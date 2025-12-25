"""Configuration for model training and distillation"""
from dataclasses import dataclass, field
from typing import Optional


# @dataclass
# class LayerwiseDistillationConfig:
#     """Layer-wise distillation configuration"""
#     use_layerwise_distillation: bool = False
#     current_training_layer: str = None #e.g. "conv_layers.0"
#     teacher_layer_name: str = None
#     learnable_projection: bool = False



@dataclass
class ParallelLayerDistillationConfig:
    """Parallel layer-wise distillation configuration"""
    teacher_layer_names: Optional[list] = None  # List of teacher layer names to distill from
    student_layer_names: Optional[list] = None  # List of student layer names to distill to
    learnable_projection: bool = False  # Whether to use learnable projections for each layer


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "cifar100"  # "mnist" or "cifar100"
    data_dir: str = "./datasets"
    imagnet_normalization: bool = False
    batch_size: int = 128
    num_workers: int = 16
    num_classes: int = 100 #change to 10 if using MNIST



@dataclass
class ViTConfig:
    """Vision Transformer configuration"""
    img_size: int = 32
    patch_size: int = 4
    stride: int = 1
    in_channels: int = 3
    embed_dim: int = 256
    depth: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    pos_embed: str = 'SymmetricPosEmbed' #'RoPE100' or 'SymmetricPosEmbed'
    # pos_embed: str = 'None-equ'
    attention_per_channel: bool = True
    group_attn_channel_pooling: bool = False


@dataclass
class PretrainedViTConfig:
    """Vision Transformer configuration"""
    # model_name: str = 'vit_b_16'
    model_name: str = 'google/vit-base-patch16-224'
    num_classes: int = 100



@dataclass
class CNNConfig:
    """Convolutional Neural Network configuration"""
    num_conv_layers: int = 5
    base_channels: int = 32


@dataclass
class TeacherModelConfig:
    """Teacher model configuration"""
    model_structure: str = "pretrained_ViT"  # "pretrained_ViT" or ?
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    pretrained_vit_config: PretrainedViTConfig = field(default_factory=PretrainedViTConfig)
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    cnn_config: CNNConfig = field(default_factory=CNNConfig)



@dataclass
class StudentModelConfig:
    """Model architecture configuration"""
    # TODO: add resnet50 option
    model_structure: str = "equ_vit"  # "equ_vit" or "equ_resnet50"
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    cnn_config: CNNConfig = field(default_factory=CNNConfig)
    


@dataclass
class TeacherTrainConfig:
    """Teacher model training configuration"""
    strategy: str = 'non_equ_train_on_GT' # 'non_equ_train_on_GT'?
    epochs: int = 10
    learning_rate: float = 3e-4
    
    # This is for testing steps
    group: str = "FlipGroup"  # "FlipGroup" or "RotationGroup"
    
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine", "exponential", "linear", "step", or "none"
    scheduler_warmup_epochs: int = 1  # Number of warmup epochs
    scheduler_step_size: int = 5  # Step size for step scheduler
    scheduler_gamma: float = 0.1  # Decay factor for exponential/step schedulers
    teacher_ckpt_path: Optional[str] = None  # Path to pre-trained teacher checkpoint
    # flip_test_images: bool = False  # If True, test on flipped images
    
@dataclass
class StudentTrainConfig:
    """Student model training configuration"""
    strategy: str = 'parallel_distillation' # 'parallel_distillation' or ?
    group: str = "FlipGroup"  # "FlipGroup" or "RotationGroup"
    epochs: int = 15
    learning_rate: float = 5e-4
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for soft loss
    # weight_decay: float = 5e-2
    weight_decay: float = 1e-1
    student_ckpt_path: Optional[str] = None  # Path to pre-trained student checkpoint
    # flip_test_images: bool = False  # If True, test on flipped images
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine", "exponential", "linear", "step", or "none"
    scheduler_warmup_epochs: int = 0  # Number of warmup epochs
    scheduler_step_size: int = 5  # Step size for step scheduler
    scheduler_gamma: float = 0.1  # Decay factor for exponential/step schedulers
    
    print_log_every_n_steps: int = 50  # Print training log every n steps
    

@dataclass
class LoggingConfig:
    """Logging configuration"""
    project_name: str = "equvariant-distillation"
    entity: Optional[str] = None  # Set to your wandb entity/team
    log_frequency: int = 50
    outputs_dir: str = "./outputs"
    wandb_name: str = "CNN_layerwise"
    wandb_mode: str = "online" 
    # every_n_train_epochs: int = 1
    # "offline" 
    # "online"


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    student_model: StudentModelConfig = field(default_factory=StudentModelConfig)
    teacher_model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    teacher_train: TeacherTrainConfig = field(default_factory=TeacherTrainConfig)
    student_train: StudentTrainConfig = field(default_factory=StudentTrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # layerwise_distillation: LayerwiseDistillationConfig = field(default_factory=LayerwiseDistillationConfig)
    parallel_layer_distillation: ParallelLayerDistillationConfig = field(default_factory=ParallelLayerDistillationConfig)
    
    
    # Training setup
    # device: str = "auto"  # Let Lightning auto-select
    device: str = "cuda"
    seed: int = 42
    precision: str = "16-mixed"  # or "16-mixed" for mixed precision
    train_teacher: bool = False  # If True, train the teacher model
    test_only: bool = False  # If True, only run testing
    
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.teacher_train.epochs > 0, "teacher_train.epochs must be positive"
        assert self.student_train.epochs > 0, "student_train.epochs must be positive"
        assert 0 <= self.student_train.alpha <= 1, "alpha must be between 0 and 1"


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
