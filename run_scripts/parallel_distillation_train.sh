
# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml


# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \

# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   teacher_model.pretrained_vit_config.model_name='WinKawaks/vit-small-patch16-224' \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt \
#   student_train.learning_rate=5e-4 \
#   student_train.scheduler_warmup_epochs=50 \
#   student_train.epochs=400 \
#   data.batch_size=64 \
#   student_model.vit_config.embed_dim=384 \
#   student_model.vit_config.n_heads=3 \
#   parallel_layer_distillation.learnable_projection=false \
#   logging.wandb_name="192_parallel_half_channel_uniform_selection_learnable_projection_ViT" \
#   student_train.student_ckpt_path="/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/double_channel/384_zero_init.ckpt"



CUDA_VISIBLE_DEVICES=0,1,3,4 python main.py \
  --config src/exp_configs/vit/parallel_distillation.yaml \
  teacher_model.pretrained_vit_config.model_name='WinKawaks/vit-small-patch16-224' \
  teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt \
  student_train.group=Rot90Group \
  student_train.learning_rate=2e-3 \
  student_train.scheduler_warmup_epochs=50 \
  student_train.epochs=500 \
  data.batch_size=32 \
  student_model.vit_config.embed_dim=384 \
  student_model.vit_config.n_heads=6 \
  parallel_layer_distillation.learnable_projection=false \
  logging.wandb_name="384_h6_parallel_zero_init_ViT" \
  student_train.student_ckpt_path="./outputs/CIFAR100/pretrained_ViT/student/initialization/rot90/double_channel/384_zero_init.ckpt"



# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   teacher_model.pretrained_vit_config.model_name='WinKawaks/vit-small-patch16-224' \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt \
#   student_train.learning_rate=5e-5 \
#   student_train.scheduler_warmup_epochs=0 \
#   student_train.epochs=200 \
#   data.batch_size=128 \
#   student_model.vit_config.embed_dim=288 \
#   student_model.vit_config.n_heads=3 \
#   parallel_layer_distillation.learnable_projection=true \
#   logging.wandb_name="288_parallel_half_channel_uniform_selection_learnable_projection_ViT" \
#     student_train.student_ckpt_path="/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/288_parallel_half_channel_uniform_selection_learnable_projection_ViT/checkpoints/best-v3.ckpt"






# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   teacher_model.pretrained_vit_config.model_name='WinKawaks/vit-small-patch16-224' \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt \
#   student_train.learning_rate=5e-4 \
#   student_train.scheduler_warmup_epochs=50 \
#   student_train.epochs=400 \
#   data.batch_size=128 \
#   student_model.vit_config.embed_dim=192 \
#   student_model.vit_config.n_heads=3 \
#   parallel_layer_distillation.learnable_projection=true \
#   logging.wandb_name="192_parallel_half_channel_uniform_selection_learnable_projection_ViT" \
#   student_train.student_ckpt_path="/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/192_zero_init_uniform_selection.ckpt"





  
  # logging.wandb_name="half_channel_wanda_learnable_projection_ViT" \
  # student_train.student_ckpt_path="/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init_wanda.ckpt"


# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   logging.wandb_name="random_double_channel_init_ViT" \
#   student_train.student_ckpt_path='' \
#   student_train.learning_rate=0.0001 \
#   student_train.epochs=300 \




# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml