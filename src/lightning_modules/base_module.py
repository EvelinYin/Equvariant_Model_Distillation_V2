import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Any, Dict, List, Optional, Union
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR, StepLR, LinearLR, SequentialLR)

from src.config import TeacherTrainConfig, StudentTrainConfig
from src.equ_lib.groups import get_group
from lightning.pytorch.utilities import grad_norm

from torch.distributions.beta import Beta
from timm.data.mixup import Mixup


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
        mixup_alpha: Optional[float] = 0.8,
        cutmix_alpha: Optional[float] = 1.0,
        cutmix_minmax: Optional[List[float]] = None,
        label_smoothing: float = 0.1,
        **kwargs
    ):
        super().__init__()
        # Saves arguments to self.hparams so they are logged to specific loggers (e.g. WandB/Tensorboard)
        self.model = model
        
        self.train_config = train_config
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.canonicalizer = None
        
        
        # # mixup and cutmix
        # self.mixup_alpha = mixup_alpha
        # self.cutmix_alpha = cutmix_alpha
        # self.label_smoothing = label_smoothing
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=cutmix_minmax,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=label_smoothing, num_classes=num_classes)
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # group
        self.group = get_group(self.train_config.group)
        self.test_accuracy_list = []
        

        self.test_accuracy_list = torch.nn.ModuleDict({
            str(i): Accuracy(task="multiclass", num_classes=num_classes)
            for i in range(self.group.elements().numel())
        })
    
        
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
        x, y = self.mixup_fn(x, y)
        output = self.forward(x)
        loss = self.compute_and_log_loss(output, y)

        return loss

    # --- Common Validation Loop ---
    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.compute_and_log_loss(output, y)

        return loss
    
    def compute_and_log_equ_tests(self, x: Any, y: Any) -> Dict[str, torch.Tensor]:
        all_logits = []
        for g in range(self.group.elements().numel()):
            transformed_x = self.group.trans(x, g)
            

            # Use canonicalizer if available
            if self.canonicalizer is not None:
                cano_x, _, _ = self.canonicalizer(transformed_x)
                # tmp_cache_cano.append(x)
            
            # Get predictions
            with torch.no_grad():
                student_logits = self.model(cano_x if self.canonicalizer is not None else transformed_x)
            
            # this is for group attn pooling
            if isinstance(student_logits, tuple):
                student_logits = student_logits[0]
            
            all_logits.append(student_logits.clone())
            
            # Update and log accuracy
            self.test_accuracy_list[str(g)](student_logits, y)
            self.log(f"test/accuracy_group{g}", self.test_accuracy_list[str(g)], on_epoch=True, on_step=False)
        
        
        all_diff = 0.0
        all_kl = 0.0
        
        p_group0 = F.softmax(all_logits[0], dim=-1)
        for i in range(1, len(all_logits)):
            all_diff += torch.norm(all_logits[0] - all_logits[i])
            
            log_p_groupi = F.log_softmax(all_logits[i], dim=-1)
            kl_val = F.kl_div(log_p_groupi, p_group0, reduction='batchmean')
            all_kl += kl_val
        # logits_diff = all_logits[0] - all_logits[1]
        all_diff /= (len(all_logits) - 1)
        all_kl /= (len(all_logits) - 1)
        self.log(f"test/logits_diff_avg_all_group", all_diff, on_epoch=True, on_step=False)
        self.log(f"test/kl_div_avg_all_group", all_kl, on_epoch=True, on_step=False)


    # --- Common Test Loop ---
    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        # pass
        x, y = batch
        
        self.compute_and_log_equ_tests(x, y)
        
        


    # --- Optimizer Configuration ---
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            eps=1e-4,
            weight_decay=self.train_config.weight_decay
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
                eta_min=1e-7
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
    
    
        
    def on_after_backward(self):
        # "2" refers to the L2 norm (Euclidean)
        # This utility calculates the norms for you
        norms = grad_norm(self, norm_type=2)
        
        # Log the dictionary of norms
        self.log_dict(norms)