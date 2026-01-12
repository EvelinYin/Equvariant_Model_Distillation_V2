"""PyTorch Lightning Module for Student Model with Knowledge Distillation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Union

from torchmetrics import Accuracy
from .base_module import BaseLightningModule

from src.config import StudentModelConfig, StudentTrainConfig, TeacherTrainConfig
from lightning.pytorch.utilities import grad_norm

from src.losses import distillation_loss
from src.equ_lib.groups import get_group

class LightningTrainingModule(BaseLightningModule):
    """Lightning Module for Student Model training with knowledge distillation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_config:  Union[TeacherTrainConfig, StudentTrainConfig],
        teacher_model: nn.Module = None,
        canonicalizer: nn.Module = None,
        num_classes: int = 10
    ):
        """
        Args:
            model_config: Model architecture configuration
            train_config: Training configuration including temperature, alpha
            teacher_model: Pre-trained teacher model
        """
        super().__init__(model=model, train_config=train_config, num_classes=num_classes)
        

        if teacher_model is None:
            self.teacher = None
        else:   
            self.teacher = teacher_model
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            self.distillation_loss = distillation_loss
        

        if canonicalizer is None:
            self.canonicalizer = None
        else:
            self.canonicalizer = canonicalizer
        
        
        self.train_config = train_config
        
        # # Save hyperparameters for logging
        # self.save_hyperparameters(ignore=["teacher"])
        
       
    
    def compute_and_log_loss(self, prediction: torch.Tensor, target: torch.Tensor, y: torch.Tensor, distill=False) -> torch.Tensor:
        
        if distill:
            loss = self.distillation_loss(prediction, target, temperature=self.train_config.temperature)
        else:
            loss = self.cross_entropy_loss(prediction, target)
        
        # Log losses
        self.log("train/total_loss", loss, on_epoch=True, on_step=False)
        
        # Log learning rate
        for param_group in self.optimizers().param_groups:
            self.log("train/learning_rate", param_group["lr"], on_epoch=True, on_step=False)
            break
        
        # Update and log accuracy
        self.train_accuracy(prediction.argmax(dim=1), y.argmax(dim=1))
        
        self.log("train/accuracy", self.train_accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    
    def training_step(self,batch, batch_idx):
        """Training step with knowledge distillation"""
        x, y = batch
        x, y = self.mixup_fn(x, y)
        
        if self.canonicalizer is not None:
            x, _, _ = self.canonicalizer(x)
        
        # Get student predictions
        logits = self.model(x)

        # Get teacher predictions (no gradient)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
            loss = self.compute_and_log_loss(logits, teacher_logits, y, distill=True)
        else:
            loss = self.compute_and_log_loss(logits, y, y)
        
        
        return loss
        

        
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        
        if self.canonicalizer is not None:
            x, _, _ = self.canonicalizer(x)
        
        with torch.no_grad():
            # Get student predictions
            logits = self.model(x)

            # Get teacher predictions
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                loss = self.distillation_loss(logits, teacher_logits, temperature=self.train_config.temperature)
            else:
                loss = self.cross_entropy_loss(logits, y)
        
        
        # Log losses
        self.log("val/total_loss", loss, on_epoch=True, on_step=False)
        
        # Update and log accuracy
        self.val_accuracy(logits, y)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        # pass
        x, y = batch
        

        
        
        self.compute_and_log_equ_tests(x, y)
        

    
    def on_after_backward(self):
        # "2" refers to the L2 norm (Euclidean)
        # This utility calculates the norms for you
        norms = grad_norm(self, norm_type=2)
        
        # Log the dictionary of norms
        self.log_dict(norms)
