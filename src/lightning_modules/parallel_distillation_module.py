"""PyTorch Lightning Module for Student Model with Knowledge Distillation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Dict, List
from .base_module import BaseLightningModule

from src.config import StudentModelConfig, StudentTrainConfig, ParallelLayerDistillationConfig
import os
from dataclasses import dataclass, asdict
from src.losses import compute_relative_error, distillation_loss
from src.utils import get_BNC_features

class StudentParallelLayerLightningModule(BaseLightningModule):
    """Lightning Module for Student Model training with knowledge distillation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_config: StudentTrainConfig,
        teacher_model: nn.Module,
        num_classes: int = 10,
        parallel_layer_distillation_config: ParallelLayerDistillationConfig = None,
    ):
        super().__init__(
            model=model,
            train_config=train_config,
            num_classes=num_classes
        )
        
        self.teacher = teacher_model
        
        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        
        self.train_config = train_config
        self.parallel_layer_distillation_config = parallel_layer_distillation_config
        
        self.teacher_layer_names = parallel_layer_distillation_config.teacher_layer_names
        self.student_layer_names = parallel_layer_distillation_config.student_layer_names
        
        self.distillation_loss = distillation_loss
        self.compute_relative_error = compute_relative_error
        self.output_proj = None

    def setup(self, stage):
        """
        Note: Initializing weights based on a random batch in DDP is risky because 
        ranks might get different batches (different weights). 
        Ideally, shapes should be inferred without data or weights broadcasted.
        """
        self.output_proj = None
        
        if self.parallel_layer_distillation_config.learnable_projection:
            self.output_proj = nn.ModuleList()
            
            # Grab one batch to infer shapes
            if stage == 'fit' or stage is None:
                # Warning: This might cause desync if shuffling is on.
                x, y = next(iter(self.trainer.datamodule.train_dataloader()))
            elif stage == 'test':
                x, y = next(iter(self.trainer.datamodule.test_dataloader()))
            
            # Using the optimized one-pass extraction for setup as well
            s_logits, t_logits, s_feats, t_feats = self._forward_and_extract_features(x)

            for layer_t, layer_s in zip(self.teacher_layer_names, self.student_layer_names):
                
                # Retrieve features from the dictionary populated by the one-pass forward
                teacher_features = t_feats.get(layer_t, t_logits if "head" in layer_t else None)
                student_features = s_feats.get(layer_s, s_logits if "head" in layer_s else None)

                if isinstance(teacher_features, tuple) and teacher_features[1] is None:
                    teacher_features = teacher_features[0]
            
                if isinstance(student_features, (list, tuple)) and not self.model.group_attn_channel_pooling:
                    # REMOVED BREAKPOINT HERE
                    for s, t in zip(student_features, teacher_features):
                        _,_,C_s = s.shape
                        _,_,C_t = t.shape 
                        self.output_proj.append(nn.Linear(int(C_s/2), C_t))
                else:
                    if "bn_layers" in layer_t:
                        teacher_features = teacher_features.permute(0, 3, 1, 2)
                    elif "head" in layer_s:
                        continue
                    
                    try:
                        reshaped_student_features = get_BNC_features(student_features)
                        reshaped_teacher_features = get_BNC_features(teacher_features)
                        _,_,C_s = reshaped_student_features.shape
                        _,_,C_t = reshaped_teacher_features.shape
                        
                        self.output_proj.append(nn.Linear(int(C_s/self.model.group.order), C_t))
                    except Exception as e:
                        # REMOVED BREAKPOINT: Raise error instead of hanging
                        raise RuntimeError(f"Shape mismatch in setup: {e}")

    def _forward_and_extract_features(self, x):
        """
        Registers hooks, runs ONE forward pass, and collects features.
        Returns: (student_logits, teacher_logits, student_features_dict, teacher_features_dict)
        """
        student_features = {}
        teacher_features = {}
        hooks = []

        # 1. Define the Hook Function
        def get_hook(storage_dict, layer_name):
            def hook(module, input, output):
                storage_dict[layer_name] = output
            return hook

        # 2. Register Student Hooks
        for layer_name in self.student_layer_names:
            # Skip "head" if it refers to the final output (logits), which we get naturally
            if "head" in layer_name and self.model.group_attn_channel_pooling:
                continue
            
            try:
                layer = self.model.get_submodule(layer_name)
                hooks.append(layer.register_forward_hook(get_hook(student_features, layer_name)))
            except AttributeError:
                print(f"Warning: Could not find layer {layer_name} in Student model")

        # 3. Register Teacher Hooks
        for layer_name in self.teacher_layer_names:
            if "head" in layer_name and self.model.group_attn_channel_pooling:
                continue
                
            try:
                layer = self.teacher.get_submodule(layer_name)
                hooks.append(layer.register_forward_hook(get_hook(teacher_features, layer_name)))
            except AttributeError:
                print(f"Warning: Could not find layer {layer_name} in Teacher model")

        # 4. Run Single Forward Pass (Student) 
        student_logits = self.model(x)
        
        # 5. Run Single Forward Pass (Teacher)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # 6. Cleanup: Remove all hooks immediately
        for h in hooks:
            h.remove()

        return student_logits, teacher_logits, student_features, teacher_features

    def _get_multi_mse_loss(self, student_features, teacher_features, layer_idx, out_proj_idx, y):
        if isinstance(teacher_features, tuple) and teacher_features[1] is None:
            teacher_features = teacher_features[0]
        
        all_losses = []
        relative_errors = []
        
        if self.model.group_attn_channel_pooling:
            if "head" in layer_idx:
                # If we passed the full logits tuple/list
                if isinstance(student_features, (tuple, list)):
                    student_features, student_logits = student_features
                else:
                    student_logits = student_features
            elif isinstance(student_features, torch.Tensor) and student_features.ndim == 3:
                student_features = torch.cat([student_features[:, :1, :], student_features[:, 2:, :]], dim=1)

        if "bn_layers" in layer_idx:
            teacher_features = teacher_features.permute(0, 3, 1, 2)
        
        if "head" in layer_idx:
            # For cross_entropy loss
            cross_entropy_loss = self.cross_entropy_loss(student_features, y)
            all_losses.append(cross_entropy_loss)

        if isinstance(student_features, (list, tuple)):
            print("Error in Line 170!!!")
            raise RuntimeError("Student features should not be a list/tuple here.")
            # REMOVED BREAKPOINT: Print warning if needed, but don't hang.
        else:
            if student_features.ndim == 2:
                loss = self.distillation_loss(student_features, teacher_features, temperature=self.train_config.temperature)
                all_losses.append(loss)
                if self.model.group_attn_channel_pooling:
                    B, _ = student_features.shape
                    pseudo_loss = self.cross_entropy_loss(student_logits, torch.zeros(B).long().to(student_features.device))
                    all_losses.append(pseudo_loss)
                    
                relative_errors.append(self.compute_relative_error(teacher_features.detach(), student_features.detach()))
                return torch.stack(all_losses), relative_errors
            else:
                reshaped_student_features = get_BNC_features(student_features)
                reshaped_teacher_features = get_BNC_features(teacher_features)
                B, N, C_s = reshaped_student_features.shape
                B, N, C_t = reshaped_teacher_features.shape
                
                if self.output_proj is not None:
                    # Check range
                    if out_proj_idx < len(self.output_proj):
                         projected_s = self.output_proj[out_proj_idx](reshaped_student_features[..., :C_s//self.model.group.order])
                    else:
                         # Fallback if indices don't match
                        raise RuntimeError(f"Output projection index {out_proj_idx} out of range.")
                        #  projected_s = student_features[:,:,:C_t]
                else:
                    projected_s = student_features[:,:,:C_t]
                
                teacher_features = reshaped_teacher_features
                all_losses.append(nn.functional.mse_loss(projected_s, teacher_features.detach()))
                
                relative_errors.append(self.compute_relative_error(teacher_features.detach(), projected_s.detach()))

            return torch.stack(all_losses), relative_errors

    def compute_and_log_loss(self, x: Any, y: Any) -> torch.Tensor:
        # OPTIMIZATION: Run forward pass ONCE
        student_logits, teacher_logits, s_feats, t_feats = self._forward_and_extract_features(x)
        
        layer_losses = []
        relative_errors = []

        # Iterate through config layers to compute loss
        for idx, (layer_t, layer_s) in enumerate(zip(self.teacher_layer_names, self.student_layer_names)):
            
            # 1. Get Teacher Features
            if "head" in layer_t and self.model.group_attn_channel_pooling:
                teacher_features = teacher_logits
            else:
                teacher_features = t_feats.get(layer_t)
                
            # 2. Get Student Features
            if "head" in layer_s and self.model.group_attn_channel_pooling:
                student_features = student_logits
            else:
                student_features = s_feats.get(layer_s)
            
            # Safety Check: If features are missing (e.g. layer name mismatch), skip or error
            if teacher_features is None or student_features is None:
                raise RuntimeError(f"Features missing for layer {layer_s} or {layer_t}")
                # # REMOVED BREAKPOINT: Fail loud or log warning
                # # print(f"Warning: Features missing for {layer_s} or {layer_t}")
                # continue

            # Compute layer-wise loss
            # We pass student_logits explicitly for the pseudo_loss calculation inside the helper
            layer_loss, relative_error = self._get_multi_mse_loss(
                student_features, teacher_features, layer_s, idx, y
            )
            
            layer_losses.extend(layer_loss)
            relative_errors.append(sum(relative_error)/len(relative_error))
            
        # Logging
        self.log("train/layer_mse_loss", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)
        
        # Accuracy Logging
        with torch.no_grad():
            # Handle tuple output if necessary
            final_logits = student_logits[0] if isinstance(student_logits, (tuple, list)) else student_logits
            self.train_accuracy(final_logits.argmax(dim=1), y.argmax(dim=1))
            self.log("train/accuracy", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False)

        if (
            self.global_rank == 0
            and self.global_step % self.train_config.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"loss = {(sum(layer_losses) / len(layer_losses)):.6f}; "
                f" relative_error = {sum(relative_errors) / len(relative_errors):.6f}; "
            )
        
        # Log losses
        self.log("train/total_loss", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)
        self.log(f"train/relative_error", sum(relative_errors) / len(relative_errors), on_epoch=True, on_step=False)
        
        for param_group in self.optimizers().param_groups:
            self.log("lr/learning_rate", param_group["lr"], on_epoch=True, on_step=False)
            break
            
        return sum(layer_losses) / len(layer_losses)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x, y = self.mixup_fn(x, y)
        loss = self.compute_and_log_loss(x, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - Optimized"""
        x, y = batch

        # OPTIMIZATION: Use the same one-pass logic for validation
        student_logits, teacher_logits, s_feats, t_feats = self._forward_and_extract_features(x)

        layer_losses = []
        relative_errors = []
        
        for idx, (layer_s, layer_t) in enumerate(zip(self.student_layer_names, self.teacher_layer_names)):    
            
            if "head" in layer_t and self.model.group_attn_channel_pooling:
                teacher_features = teacher_logits
            else:
                teacher_features = t_feats.get(layer_t)

            if "head" in layer_s and self.model.group_attn_channel_pooling:
                student_features = student_logits
            else:
                student_features = s_feats.get(layer_s)

            if teacher_features is not None and student_features is not None:
                layer_mse, relative_error = self._get_multi_mse_loss(
                    student_features, teacher_features, layer_s, idx, y
                )
                layer_losses.extend(layer_mse)
                relative_errors.append(sum(relative_error) / len(relative_error))
 
        self.log(f"val/mse", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)
        
        if self.model.linear_pooling:
            reg_loss = 0.1 * (torch.norm(self.model.linear_pooling_layer.linear.weight - self.W_ls.to(x.device))**2 +
                            torch.norm(self.model.linear_pooling_layer.linear.bias - self.b_ls.to(x.device))**2 )
            layer_losses.append(reg_loss)
            self.log("val/reg_loss", reg_loss, on_epoch=True, on_step=False)
            
        self.log(f"val/relative_error", sum(relative_errors) / len(relative_errors), on_epoch=True, on_step=False)
        
        # Accuracy
        final_logits = student_logits[0] if isinstance(student_logits, (tuple, list)) else student_logits
        self.val_accuracy(final_logits, y)
        self.log("val/accuracy", self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
        return sum(layer_losses) / len(layer_losses)

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict.keys() if "output_proj" in k]
        for k in keys_to_remove:
            del state_dict[k]
            print(f"Deleted key from checkpoint: {k}")