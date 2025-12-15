"""PyTorch Lightning Module for Student Model with Knowledge Distillation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LinearLR, SequentialLR
from torchmetrics import Accuracy
from typing import Any
from .base_module import BaseLightningModule

from src.config import StudentModelConfig, StudentTrainConfig, ParallelLayerDistillationConfig
import os
from lightning.pytorch.utilities import grad_norm
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
        """
        Args:
            model_config: Model architecture configuration
            train_config: Training configuration including temperature, alpha
            teacher_model: Pre-trained teacher model
        """
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

    
    def setup(self, stage):
        self.output_proj = None
        
        if self.parallel_layer_distillation_config.learnable_projection:
            self.output_proj = nn.ModuleList()
            
            for layer_t, layer_s in zip(self.teacher_layer_names, self.student_layer_names):
                x, y = next(iter(self.trainer.datamodule.train_dataloader()))
                
                with torch.no_grad():
                    teacher_features = self._get_layer_features(self.teacher, x, layer_t, False)
                    student_features = self._get_layer_features(self.model, x, layer_s, False)
            
            # TODO: handle different feature shapes and for different architectures
            # if "patch_embed" in self.layerwise_config.current_training_layer or \
            #     "dec_block" in self.layerwise_config.current_training_layer:
            #     student_features = student_features[0]
            #     teacher_features = teacher_features[0]
                if isinstance(teacher_features, tuple) and teacher_features[1] is None:
                    teacher_features = teacher_features[0]
            
                if isinstance(student_features, list) or isinstance(student_features, tuple):
                    breakpoint()
                    for s, t in zip(student_features, teacher_features):
                        _,_,C_s = s.shape
                        _,_,C_t = t.shape 
                        self.output_proj.append(nn.Linear(int(C_s/2), C_t))
                        # self.output_proj.append(nn.Linear(s.shape[-1], t.shape[-1]))
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
                        
                        breakpoint()
                        self.output_proj.append(nn.Linear(int(C_s/self.model.group.order), C_t))
                            # self.output_proj.append(nn.Linear(int(C_s/2), C_t))
                    except:
                        breakpoint()

        # if self.student.linear_pooling:
        #     self.W_ls = self.student.linear_pooling_layer.linear.weight.detach().clone()
        #     self.b_ls = self.student.linear_pooling_layer.linear.bias.detach().clone()
    

    
    def _get_layer_features(self, model: nn.Module, context: Any, layer_idx: str, requires_grad: bool) -> Any:
        """Extract features from a specific layer of the model."""
        features = []
        
        if "head" in layer_idx and self.model.group_attn_channel_pooling:
            return model(context)

        def hook_fn(module, input, output):
            features.append(output)

        # Register hook on the specific layer
        handle = model.get_submodule(layer_idx).register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad() if model == self.teacher else torch.enable_grad():
            _ = model(context)
        
        # Remove hook
        handle.remove()

        return features[0]
    
    
    
    def _get_multi_mse_loss(self, student_features, teacher_features, layer_idx, out_proj_idx):
        # if self.train_cfg.current_training_layer == "all":
        
        if isinstance(teacher_features, tuple) and teacher_features[1] is None:
            teacher_features = teacher_features[0]
        
        # breakpoint()
        if self.model.group_attn_channel_pooling:
            if "head" in layer_idx:
                    student_features, student_logits = student_features
            elif student_features.ndim == 3:
                student_features = torch.cat([student_features[:, :1, :], student_features[:, 2:, :]], dim=1)

        if "bn_layers" in layer_idx:
            teacher_features = teacher_features.permute(0, 3, 1, 2)
        
        
        
        all_losses = []
        relative_errors = []
      

        if isinstance(student_features, list) or isinstance(student_features, tuple):
            breakpoint()
            # for s, t, proj in zip(student_features, teacher_features, self.output_proj):
            #     projected_s = proj(s[..., :s.shape[-1]//2])
            #     loss = nn.functional.mse_loss(projected_s, t.detach())
            #     all_losses.append(loss)
            #     relative_errors.append(self._compute_relative_error(t.detach(), projected_s.detach()))
        else:
            if student_features.ndim == 2:
                loss = self.distillation_loss(student_features, teacher_features, temperature=self.train_config.temperature)
                all_losses.append(loss)
                if self.model.group_attn_channel_pooling:
                    B, _ = student_features.shape
                    pseudo_loss = self.cross_entropy_loss(student_logits, torch.zeros(B).long().to(student_logits.device))
                    all_losses.append(pseudo_loss)
                relative_errors.append(self.compute_relative_error(teacher_features.detach(), student_features.detach()))
                return torch.stack(all_losses), relative_errors
            else:
                reshaped_student_features = get_BNC_features(student_features)
                reshaped_teacher_features = get_BNC_features(teacher_features)
                B, N, C_s = reshaped_student_features.shape
                B, N, C_t = reshaped_teacher_features.shape
                
                if self.output_proj is not None:
                    projected_s = self.output_proj[out_proj_idx](reshaped_student_features[..., :C_s//self.model.group.order])
                else:
                    projected_s = student_features[:,:,:C_t]
                
                teacher_features = reshaped_teacher_features
                all_losses.append(nn.functional.mse_loss(projected_s, teacher_features.detach()))
                
                relative_errors.append(self.compute_relative_error(teacher_features.detach(), projected_s.detach()))
                # relative_errors.append(self.compute_relative_error(flipped_t.detach(), projected_s2.detach()))
        

            return torch.stack(all_losses), relative_errors

    
    
    def compute_and_log_loss(self, x: Any, y: Any) -> torch.Tensor:
        layer_losses = []
        relative_errors = []
        for idx, (layer_t, layer_s) in enumerate(zip(self.teacher_layer_names, self.student_layer_names)):
            
            try:
                # Get teacher features for the current layer
                teacher_features = self._get_layer_features(self.teacher, x, layer_t, False)
            except Exception as e:
                breakpoint()
            
            # Get student features for the current layer
            student_features = self._get_layer_features(self.model, x, layer_s, True)
            
            # Compute layer-wise loss (MSE between teacher and student features)
            layer_loss, relative_error = self._get_multi_mse_loss(student_features, teacher_features, layer_s, idx)
            # layer_loss = layer_loss.mean()
            relative_error = sum(relative_error)/len(relative_error)
            
            # layer_losses.append(layer_loss)
            layer_losses.extend(layer_loss)
            relative_errors.append(relative_error)
        
        self.log("train/layer_mse_loss", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)

        
        # Log losses
        self.log("train/total_loss", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)
        self.log(f"train/relative_error", sum(relative_errors) / len(relative_errors), on_epoch=True, on_step=False)
        # self.log("train/hard_loss", hard_loss, on_epoch=True, on_step=False)
        # self.log("train/soft_loss", soft_loss, on_epoch=True, on_step=False)
        
        # Log learning rate
        for param_group in self.optimizers().param_groups:
            self.log("lr/learning_rate", param_group["lr"], on_epoch=True, on_step=False)
            break
        
        # # Update and log accuracy
        with torch.no_grad():
            student_logits = self.model(x)
            if self.model.group_attn_channel_pooling:
                student_logits = student_logits[0]
                
            self.train_accuracy(student_logits, y)
            self.log("train/accuracy", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
        if (
            self.global_rank == 0
            and self.global_step % self.train_config.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"loss = {(sum(layer_losses) / len(layer_losses)):.6f}; "
                f" relative_error = {relative_error:.6f}; "
            )
            
        return sum(layer_losses) / len(layer_losses)
    
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # 1. Compute loss and metrics
        x, y = batch
        loss = self.compute_and_log_loss(x, y)

        return loss

    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch


        layer_losses = []
        relative_errors = []
        for idx, (layer_s, layer_t) in enumerate(zip(self.student_layer_names, self.teacher_layer_names)):    
                # try:
                    # Layer-specific validation
                    teacher_features = self._get_layer_features(self.teacher, x, layer_t, False)
                    student_features = self._get_layer_features(self.model, x, layer_s, True)
                    

                    if teacher_features is not None and student_features is not None:
                        layer_mse, relative_error = self._get_multi_mse_loss(student_features, teacher_features, layer_s, idx)
                        layer_losses.extend(layer_mse)
                        relative_errors.append(sum(relative_error) / len(relative_error))
                # except Exception as e:
                #     print("bugg??")
                #     print(e)
                #     breakpoint()     
        # breakpoint()
        self.log(f"val/mse", sum(layer_losses) / len(layer_losses), on_epoch=True, on_step=False)
        
        # reg loss
        if self.model.linear_pooling:
            reg_loss = 0.1 * (torch.norm(self.model.linear_pooling_layer.linear.weight - self.W_ls.to(x.device))**2 +
                            torch.norm(self.model.linear_pooling_layer.linear.bias - self.b_ls.to(x.device))**2 )
            layer_losses.append(reg_loss)
            self.log("val/reg_loss", reg_loss, on_epoch=True, on_step=False)
            
        self.log(f"val/relative_error", sum(relative_errors) / len(relative_errors), on_epoch=True, on_step=False)
        
        if (
            self.global_rank == 0
            and self.global_step % self.train_config.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"loss = {layer_mse.mean():.6f}; "
                f" relative_error = {sum(relative_error) / len(relative_error):.6f}"
            )
                                            

        def inject_forward_output(model, inject_layer_name, injected_tensor, teacher_features):
            """
            Replaces the output of a given layer in a model during forward pass.
            """
            student_x = injected_tensor
            

            if self.output_proj is not None:
                    with torch.no_grad():
                            # breakpoint()
                            if isinstance(student_x, list) or isinstance(student_x, tuple):
                                for s in student_x:
                                    B,N,C = s.shape
                                    student_x = [self.output_proj[i](s[:, :, :s.shape[-1]//2]) for i, s in enumerate(student_x)]
                                    injected_tensor = student_x
                            else:
                                if student_x.ndim == 5:
                                    B,_,_,H,W = student_x.shape
                                    student_x = student_x.reshape(B, -1, H, W)
                                    C = student_x.size(1)
                                    student_x = self.output_proj[0](student_x.permute(0, 2, 3, 1)[..., :C//2]).permute(0, 3, 1, 2)
                                    
                                elif student_x.ndim == 4:
                                    B,C,H,W = student_x.shape
                                    student_x = self.output_proj[0](student_x.permute(0, 2, 3, 1)[..., :C//2]).permute(0, 3, 1, 2)

                                elif student_x.ndim == 3:
                                    B,N,C = student_x.shape
                                    student_x = self.output_proj[0](student_x[:, :, :C//2])

                                if "bn_layers" in inject_layer_name:
                                    student_x = student_x.permute(0, 2, 3, 1)
                                injected_tensor = student_x
            else:
                    if student_x.ndim == 3:
                        _, _, C_s = student_x.shape
                        _, _, C_t = teacher_features.shape
                        injected_tensor = student_x[:,:,:C_t]

                    else:
                        breakpoint()
                        # injected_tensor = student_x
            
            def hook_fn(module, input, output):
                return injected_tensor
            
            
            handle = None
            for name, module in model.named_modules():
                if name == inject_layer_name:
                    handle = module.register_forward_hook(hook_fn)
                    break

            if handle is None:
                raise ValueError(f"Layer {inject_layer_name} not found in model.")

            return handle  # So you can remove the hook after forward

     
        # if layer_idx != "all":
        #     handle = inject_forward_output(
        #         self.teacher,
        #         self.layerwise_config.teacher_layer_name,
        #         student_features,
        #         teacher_features
        #     )

        #     logits = self.teacher(x)

        #     handle.remove()  # Remove the hook after forward pass
        
        teacher_logits = self.teacher(x)
        student_logits = self.model(x)
        
        if self.model.group_attn_channel_pooling:
            student_logits = student_logits[0]
        
            # Get teacher predictions (no gradient) using injected features from student
            # logits = self.teacher(x)

        
        # Compute loss
        # loss, hard_loss, soft_loss = self.distillation_loss(student_logits, teacher_logits, y)
        # loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        # Log losses
        # self.log("val/total_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # Update and log accuracy
        # self.val_accuracy(logits, y)
        self.val_accuracy(student_logits, y)
        self.log("val/accuracy", self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
        return layer_mse.mean()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch


        # x = torch.flip(x, dims=[-1])
        
        # Get predictions
        with torch.no_grad():
            # teacher_logits = self.teacher(x)
            student_logits = self.model(x)
            
            if self.model.group_attn_channel_pooling:
                student_logits = student_logits[0]
        
        # Compute loss
        # loss, hard_loss, soft_loss = self.distillation_loss(student_logits, teacher_logits, y)
        # loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        loss = self.cross_entropy_loss(student_logits, y)
        
        # Log losses
        self.log("test/total_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        # self.log("test/hard_loss", hard_loss, on_epoch=True, on_step=False)
        # self.log("test/soft_loss", soft_loss, on_epoch=True, on_step=False)
        
        # Update and log accuracy
        self.test_accuracy(student_logits, y)
        self.log("test/accuracy", self.test_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss
    
    
    def on_after_backward(self):
        # "2" refers to the L2 norm (Euclidean)
        # This utility calculates the norms for you
        norms = grad_norm(self, norm_type=2)
        
        # Log the dictionary of norms
        self.log_dict(norms)
    
    
    def on_load_checkpoint(self, checkpoint):
        # Get the state_dict from the checkpoint
        state_dict = checkpoint["state_dict"]
        
        # create a list of keys to remove (e.g., the classifier weights)
        # Note: In PL, keys usually have the same name as your attributes
        keys_to_remove = [k for k in state_dict.keys() if "output_proj" in k]
        
        for k in keys_to_remove:
            del state_dict[k]
            print(f"Deleted key from checkpoint: {k}")