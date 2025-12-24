"""Main training script using PyTorch Lightning with wandb logging"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import torch.nn as nn

from src.utils import setup_training_directories, load_model_checkpoint
from src.config import Config, get_default_config
from src.models import get_model
from src.data_modules import get_datamodule
from src.lightning_modules import get_lightining_modules


from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from torch.cuda.amp import GradScaler
import gc 




def test_student(config: Config, data_module: pl.LightningDataModule,
                  student_model: nn.Module, teacher_model: nn.Module = None,
                  ):
    """Train student model with knowledge distillation using Lightning Trainer"""
    
   
    
    
    # Setup
    pl.seed_everything(config.seed)
    
    
    student_module = get_lightining_modules(
        strategy=config.student_train.strategy,
        config=config,
        model=student_model,
        teacher_model=teacher_model,
    )

    
     # Load pre-trained student checkpoint if provided
    student_ckpt_path = config.student_train.student_ckpt_path
    if student_ckpt_path is not None:
            # 2. Load the state dict manually
            checkpoint = torch.load(student_ckpt_path, map_location='cpu')
            student_module.load_state_dict(checkpoint['state_dict'])
    
    
    # Initialize wandb
    wandb.login()
    
    
    # Setup wandb logger for student
    wandb_logger_student = WandbLogger(
        project=config.logging.project_name,
        entity=config.logging.entity,
        name=f"student_distillation_{config.logging.wandb_name}",
        log_model=False,
        mode=config.logging.wandb_mode,
        save_dir=config.logging.outputs_dir
    )
    

    
    
    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger_student,
        max_epochs=1,
        accelerator=config.device,
        devices="auto",
        precision=config.precision, 
        enable_progress_bar=True,
        # log_every_n_steps=config.logging.log_frequency,
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm" 
    )
    
    
  

    # 3. Run test, but set ckpt_path=None so Trainer doesn't try to reload it strictly
    trainer.test(student_module, datamodule=data_module)
    
    
    
    
    
def test(config: Config = None, set_train_teacher=False):
    """Main training pipeline"""
    if config is None:
        config = get_default_config()
    
    setup_training_directories(config)
    
    
    data_module = get_datamodule(config.data)
    
    model = get_model(name=config.student_model.model_structure, config=config.student_model)
    teacher_model = get_model(name=config.teacher_model.model_structure, config=config.teacher_model)
    teacher_model = load_model_checkpoint(teacher_model, config.teacher_train.teacher_ckpt_path)
    test_student(config, student_model=model, teacher_model=teacher_model, data_module=data_module)


    # teacher_module = None
    
    # if train_equ_w_gt:
    #     # Train student with ground truth labels only (no distillation)
    #     student_module = train_student(config, teacher_module=None, student_ckpt_path=student_ckpt_path, train_equ_w_gt=True)
    #     return
    
    
    # breakpoint()

    # Train student with distillation
    # student_module = train_student(config, teacher_module, student_ckpt_path=student_ckpt_path)
    