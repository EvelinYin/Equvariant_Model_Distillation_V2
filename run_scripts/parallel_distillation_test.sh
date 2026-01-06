

# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/checkpoints/best-v1.ckpt \
#   precision=32 \
#   data.num_workers=0 \
#   test_only=true \
#   student_model.vit_config.group_attn_channel_pooling=true


# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/checkpoints/best-v7.ckpt \
#   precision=32 \
#   data.num_workers=0 \
#   student_model.vit_config.embed_dim=384 \
#   parallel_layer_distillation.learnable_projection=true \
#   student_model.vit_config.group_attn_channel_pooling=true \
#   test_only=true \
#   logging.wandb_mode="offline"


# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/avg_pool/checkpoints/best.ckpt \
#   precision=32 \
#   data.num_workers=0 \
#   test_only=true \
#   logging.wandb_mode="offline"
#   # student_model.vit_config.group_attn_channel_pooling=true \


# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillationrandom_double_channel_init_ViT/checkpoints/best-v9.ckpt\
#   precision=64 \
#   data.num_workers=0 \
#   test_only=true \
#   logging.wandb_mode="offline" \
#   student_model.vit_config.group_attn_channel_pooling=true


CUDA_VISIBLE_DEVICES=0 python main.py \
  --config src/exp_configs/vit/parallel_distillation.yaml \
  student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/half_channel_wanda_learnable_projection_ViT/checkpoints/best.ckpt \
  student_model.vit_config.embed_dim=384 \
  precision=64 \
  data.num_workers=0 \
  test_only=true \
  logging.wandb_mode="offline" \
  student_model.vit_config.group_attn_channel_pooling=true

# 

# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml