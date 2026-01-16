"""Main training script using PyTorch Lightning with wandb logging"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import torch.nn as nn

from src.utils import setup_training_directories, load_model_checkpoint, clean_state_dict
from src.config import Config, get_default_config
from src.models import get_model
from src.data_modules import get_datamodule
from src.lightning_modules import get_lightining_modules


from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from torch.cuda.amp import GradScaler
import gc 



def train_teacher(config: Config, data_module: pl.LightningDataModule,
                  teacher_model: nn.Module = None,
                  ):

    # Load pre-trained checkpoint if provided
    teacher_ckpt_path = config.teacher_train.teacher_ckpt_path
    if teacher_ckpt_path is not None and teacher_ckpt_path != '':
        teacher_model = load_model_checkpoint(teacher_model, teacher_ckpt_path)

    
    
    # Setup
    pl.seed_everything(config.seed)
    
    
    teacher_module = get_lightining_modules(
        strategy=config.teacher_train.strategy,
        config=config,
        model=teacher_model
    )
    
    # set checkpoints directory
    config.logging.outputs_dir = f'./outputs/{config.teacher_train.group}/' \
                                    + f'{config.data.dataset_name}/' \
                                    + 'teacher/' \
                                    + f'{config.teacher_model.model_structure}/' \
                                    + f'{config.teacher_train.strategy}/' \
                                    + f'{config.logging.wandb_name}'
    
    # Initialize wandb
    wandb.login()
    
    
    # Setup wandb logger
    wandb_logger_teacher = WandbLogger(
        project=config.logging.project_name,
        entity=config.logging.entity,
        name=f"teacher_distillation_{config.logging.wandb_name}",
        log_model=False,
        mode=config.logging.wandb_mode,
        save_dir=config.logging.outputs_dir
    )
    
    # Log config to wandb
    wandb_logger_teacher.log_hyperparams(config.to_dict() if hasattr(config, 'to_dict') else vars(config))
    
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.logging.outputs_dir, "checkpoints"),
        filename='best',
        save_top_k=1,
        monitor="val/accuracy",
        mode="max",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/accuracy",
        patience=50,
        mode="max",
        verbose=True,
    )
    
    
    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger_teacher,
        callbacks=[checkpoint_callback, early_stop_callback],
        # callbacks=[checkpoint_callback],
        max_epochs=config.teacher_train.epochs,
        accelerator=config.device,
        precision=config.precision, 
        # plugins=[MixedPrecisionPlugin(precision=config.precision, device=config.device, scaler=scaler)],
        enable_progress_bar=True,
        # log_every_n_steps=config.logging.log_frequency,
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm" 
    )
    
    if config.test_only:
        print("Test only mode enabled. Skipping training.")
        trainer.test(teacher_module, datamodule=data_module, ckpt_path=None)
    
    else:
    
        # Train
        if (teacher_ckpt_path is not None and teacher_ckpt_path != '' ) \
            and 'pytorch-lightning_version' in torch.load(teacher_ckpt_path, map_location='cpu'):
            print("Resuming Trainer state (Optimizer, Scheduler, Epoch)...")
            # Passing ckpt_path here restores the FULL training state
            trainer.fit(teacher_module, datamodule=data_module, ckpt_path=teacher_ckpt_path)
        else:
            trainer.fit(teacher_module, datamodule=data_module)
        
        # Test
        # 1. Get the path to the best checkpoint
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"Loading best checkpoint from: {best_path}")
        
        
        # 2. Load the state dict manually
        checkpoint = torch.load(best_path)
        teacher_module.load_state_dict(checkpoint['state_dict'])

        # 3. Run test, but set ckpt_path=None so Trainer doesn't try to reload it strictly
        trainer.test(teacher_module, datamodule=data_module, ckpt_path=None)
        
    # Close wandb run
    wandb_logger_teacher.experiment.finish()
    
    
    





def train_student(config: Config, data_module: pl.LightningDataModule,
                  student_model: nn.Module, teacher_model: nn.Module = None,
                  ):
    """Train student model with knowledge distillation using Lightning Trainer"""
    # Load pre-trained student checkpoint if provided
    student_ckpt_path = config.student_train.student_ckpt_path
    if student_ckpt_path is not None and student_ckpt_path != '':
        # checkpoint = torch.load(student_ckpt_path, map_location='cpu')
        # if 'state_dict' in checkpoint:
        #     # This is a Lightning checkpoint - filter the state_dict
        #     state_dict = checkpoint['state_dict']
            
        #     # Filter to only student model weights
        #     student_state_dict = {
        #         k.replace('model.', '', 1): v  # Remove 'model.' prefix if it exists
        #         for k, v in state_dict.items() 
        #         if not k.startswith('teacher.')
        #     }
            
        #     # Load into student model
        #     student_model.load_state_dict(student_state_dict, strict=False)
        #     print(f"Loaded student model weights from {student_ckpt_path}")
        # else:
            # This is a regular model checkpoint
        student_model = load_model_checkpoint(student_model, student_ckpt_path)

    
    
    # Setup
    pl.seed_everything(config.seed)
    
    
    student_module = get_lightining_modules(
        strategy=config.student_train.strategy,
        config=config,
        model=student_model,
        teacher_model=teacher_model,
    )
    
    # set checkpoints directory
    config.logging.outputs_dir = f'./outputs/{config.student_train.group}' \
                                    + f'{config.data.dataset_name}/' \
                                    + 'student/' \
                                    + f'{config.student_model.model_structure}/' \
                                    + f'{config.student_train.strategy}/' \
                                    + f'{config.logging.wandb_name}'
    
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
    
    # Log config to wandb
    wandb_logger_student.log_hyperparams(config.to_dict() if hasattr(config, 'to_dict') else vars(config))
    
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.logging.outputs_dir, "checkpoints"),
        filename='best',
        save_top_k=1,
        monitor="val/accuracy",
        mode="max",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/accuracy",
        patience=50,
        mode="max",
        verbose=True,
    )
    
    
    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger_student,
        callbacks=[checkpoint_callback, early_stop_callback],
        # callbacks=[checkpoint_callback],
        max_epochs=config.student_train.epochs,
        accelerator=config.device,
        precision=config.precision, 
        # plugins=[MixedPrecisionPlugin(precision=config.precision, device=config.device, scaler=scaler)],
        enable_progress_bar=True,
        # log_every_n_steps=config.logging.log_frequency,
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm" 
    )
    
    if config.test_only:
        print("Test only mode enabled. Skipping training.")
        trainer.test(student_module, datamodule=data_module, ckpt_path=None)
        # trainer.test(student_module, datamodule=data_module, ckpt_path=student_ckpt_path)
        
    
    else:
        # Train
        if (student_ckpt_path is not None and student_ckpt_path != '' and config.student_train.resume_training) \
            and 'pytorch-lightning_version' in torch.load(student_ckpt_path, map_location='cpu'):
            print("Resuming Trainer state (Optimizer, Scheduler, Epoch)...")
            # Passing ckpt_path here restores the FULL training state
            trainer.fit(student_module, datamodule=data_module, ckpt_path=student_ckpt_path)
        else:
            trainer.fit(student_module, datamodule=data_module)
        
        # Test
        # 1. Get the path to the best checkpoint
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"Loading best checkpoint from: {best_path}")
        
        
        # 2. Load the state dict manually
        checkpoint = torch.load(best_path)
        student_module.load_state_dict(checkpoint['state_dict'])

        # 3. Run test, but set ckpt_path=None so Trainer doesn't try to reload it strictly
        trainer.test(student_module, datamodule=data_module, ckpt_path=None)
        
    # Close wandb run
    wandb_logger_student.experiment.finish()
    
    
    
    
def train(config: Config = None):
    """Main training pipeline"""
    if config is None:
        config = get_default_config()
    
    setup_training_directories(config)
    
    
    data_module = get_datamodule(config.data)
    

    
    if not config.train_teacher:
        model = get_model(name=config.student_model.model_structure, config=config.student_model, group=config.student_train.group)
        teacher_model = get_model(name=config.teacher_model.model_structure, config=config.teacher_model)
        if config.teacher_train.teacher_ckpt_path is not None and config.teacher_train.teacher_ckpt_path != '':
            teacher_model = load_model_checkpoint(teacher_model, config.teacher_train.teacher_ckpt_path)
        train_student(config, student_model=model, teacher_model=teacher_model, data_module=data_module)
    
    else:
        model = get_model(name=config.teacher_model.model_structure, config=config.teacher_model)
        train_teacher(config, teacher_model=model, data_module=data_module)


    # teacher_module = None
    
    # if train_equ_w_gt:
    #     # Train student with ground truth labels only (no distillation)
    #     student_module = train_student(config, teacher_module=None, student_ckpt_path=student_ckpt_path, train_equ_w_gt=True)
    #     return
    
    
    # breakpoint()

    # Train student with distillation
    # student_module = train_student(config, teacher_module, student_ckpt_path=student_ckpt_path)
    