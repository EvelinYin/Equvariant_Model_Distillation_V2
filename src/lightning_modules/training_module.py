"""PyTorch Lightning Module for Student Model with Knowledge Distillation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .base_module import BaseLightningModule
from config import StudentModelConfig, StudentTrainConfig, TeacherTrainConfig
from lightning.pytorch.utilities import grad_norm

from src.losses import distillation_loss


class LightningTrainingModule(BaseLightningModule):
    """Lightning Module for Student Model training with knowledge distillation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_config:  Union[TeacherTrainConfig, StudentTrainConfig],
        teacher_model: nn.Module,
        num_classes: int = 10
    ):
        """
        Args:
            model_config: Model architecture configuration
            train_config: Training configuration including temperature, alpha
            teacher_model: Pre-trained teacher model
        """
        super().__init__()
        

        if teacher_model is None:
            self.teacher = None
        else:   
            self.teacher = teacher_model
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            self.distillation_loss = distillation_loss
        
        self.train_config = train_config
        
        # # Save hyperparameters for logging
        # self.save_hyperparameters(ignore=["teacher"])
    
    def training_step(self,batch, batch_idx):
        """Training step with knowledge distillation"""
        x, y = batch
        


        # Get teacher predictions (no gradient)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
        else:
            teacher_logits = None

        
        # Get student predictions
        student_logits = self.student(x)
        

        
        # Compute distillation loss
        # loss, hard_loss, soft_loss = self.distillation_loss(student_logits, teacher_logits, y)
        # loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        if self.teacher is not None:
            teacher_probs = F.softmax(teacher_logits, dim=1)
            teacher_preds = torch.argmax(teacher_probs, dim=1)
            loss = self.cross_entropy_loss(student_logits, teacher_preds)
        else:
            loss = self.cross_entropy_loss(student_logits, y)
        
        
        
        # Log losses
        self.log("train/total_loss", loss, on_epoch=True, on_step=False)
        # self.log("train/hard_loss", hard_loss, on_epoch=True, on_step=False)
        # self.log("train/soft_loss", soft_loss, on_epoch=True, on_step=False)
        
        # Log learning rate
        for param_group in self.optimizers().param_groups:
            self.log("train/learning_rate", param_group["lr"], on_epoch=True, on_step=False)
            break
        
        # Update and log accuracy
        self.train_accuracy(student_logits, y)
        self.log("train/accuracy", self.train_accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            if self.teacher is not None:
                teacher_logits = self.teacher(x)
            else:
                teacher_logits = None
            student_logits = self.student(x)
        
        # breakpoint()
        
        # Compute loss
        # loss, hard_loss, soft_loss = self.distillation_loss(student_logits, teacher_logits, y)
        # loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        if self.teacher is not None:
            teacher_probs = F.softmax(teacher_logits, dim=1)
            teacher_preds = torch.argmax(teacher_probs, dim=1)
            loss = self.cross_entropy_loss(student_logits, teacher_preds)
        else:
            loss = self.cross_entropy_loss(student_logits, y)
        
        
        # Log losses
        self.log("val/total_loss", loss, on_epoch=True, on_step=False)
        # self.log("val/hard_loss", hard_loss, on_epoch=True, on_step=False)
        # self.log("val/soft_loss", soft_loss, on_epoch=True, on_step=False)
        
        # Update and log accuracy
        self.val_accuracy(student_logits, y)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        
        # Get predictions
        with torch.no_grad():
            if self.teacher is not None:
                teacher_logits = self.teacher(x)
                student_logits = self.student(x)
                loss = self.distillation_loss(student_logits, teacher_logits, y)
                
            else:
                student_logits = self.student(x)
                loss = self.cross_entropy_loss(student_logits, y)
        
        # Compute loss
        # loss, hard_loss, soft_loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        # Log losses
        self.log("test/total_loss", loss, on_epoch=True, on_step=False)
        # self.log("test/hard_loss", hard_loss, on_epoch=True, on_step=False)
        # self.log("test/soft_loss", soft_loss, on_epoch=True, on_step=False)
        
        # Update and log accuracy
        self.test_accuracy(student_logits, y)
        self.log("test/accuracy", self.test_accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    
    def on_after_backward(self):
        # "2" refers to the L2 norm (Euclidean)
        # This utility calculates the norms for you
        norms = grad_norm(self, norm_type=2)
        
        # Log the dictionary of norms
        self.log_dict(norms)
