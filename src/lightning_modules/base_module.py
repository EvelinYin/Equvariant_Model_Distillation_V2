import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Any, Dict, List, Optional, Union
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR, StepLR, LinearLR, SequentialLR)
from src.config import TeacherTrainConfig, StudentTrainConfig

class BaseLightningModule(pl.LightningModule):
    """
    Base LightningModule that handles common logic for:
    - Optimizer/Scheduler configuration
    - Basic logging
    - Model saving/loading hooks
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_config: Union[TeacherTrainConfig, StudentTrainConfig],
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__()
        # Saves arguments to self.hparams so they are logged to specific loggers (e.g. WandB/Tensorboard)
        self.model = model
        
        self.train_config = train_config
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generic forward pass. Child classes should override this
        or rely on a self.net nn.Module defined in their __init__.
        """
        return self.model(x)

    def compute_and_log_loss(self, prediction: Any, target: Any) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for a batch.
        Child classes MUST implement this.
        Returns a dictionary containing 'loss' and any other metrics to log.
        """
        raise NotImplementedError("Child class must implement compute_loss()")
    

    # --- Common Training Loop ---
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # 1. Compute loss and metrics
        x, y = batch
        output = self.forward(x)
        loss = self.compute_and_log_loss(output, y)

        return loss

    # --- Common Validation Loop ---
    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.compute_and_log_loss(output, y)

        return loss

    # --- Common Test Loop ---
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.compute_and_log_loss(output, y)

        return loss

    # --- Optimizer Configuration ---
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            eps=1e-4
            # weight_decay=self.train_config.weight_decay
        )
        
        # Get total number of steps
        total_epochs = self.train_config.epochs
        steps_per_epoch = self.trainer.estimated_stepping_batches // total_epochs if self.trainer else 1
        
        scheduler_config = {
            "scheduler": None,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/total_loss"
        }
        

        
        
        scheduler_type = self.train_config.scheduler_type.lower()
        
        if scheduler_type == "none":
            # No scheduler
            return optimizer
        
        elif scheduler_type == "cosine":
            # Cosine annealing with warmup
            warmup_epochs = self.train_config.scheduler_warmup_epochs

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-6
            )

            
        elif scheduler_type == "exponential":
            # Exponential decay
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.train_config.scheduler_gamma
            )

            
        elif scheduler_type == "step":
            # Step decay
            scheduler = StepLR(
                optimizer,
                step_size=self.train_config.scheduler_step_size,
                gamma=self.train_config.scheduler_gamma
            )

            
        elif scheduler_type == "linear":
            # Linear decay
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_epochs
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        if self.train_config.scheduler_warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.train_config.scheduler_warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.train_config.scheduler_warmup_epochs]
            )


        scheduler_config["scheduler"] = scheduler
        
        
        # Return scheduler config
        if scheduler_config["scheduler"] is not None:
            return [optimizer], [scheduler_config]
        return optimizer
    