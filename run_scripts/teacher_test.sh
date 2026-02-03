
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   precision=32 \
#   data.num_workers=0 \
#   train_teacher=true \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt \
#   test_only=true \
#   logging.wandb_mode="offline" \


# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   precision=64 \
#   data.num_workers=0 \
#   train_teacher=true \
#   teacher_train.teacher_ckpt_path=./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init_wanda.ckpt \
#   test_only=true \
#   logging.wandb_mode="offline" \


CUDA_VISIBLE_DEVICES=2 python main.py \
  --config src/exp_configs/vit/teacher_finetune.yaml \
  precision=64 \
  data.num_workers=8 \
  train_teacher=true \
  teacher_model.pretrained_vit_config.model_name='WinKawaks/vit-small-patch16-224' \
  teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/Rot45Group/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_rot45aug_vit_small_weight_selection/checkpoints/best.ckpt \
  teacher_train.group="Rot45Group" \
  test_only=true \
  logging.wandb_mode="online" \
  logging.wandb_name="test_teacher_rot45_aug_finetune_winKawaks_vit_small" \
# /home/yin178/Equvariant_Model_Distillation_V2/outputs/Rot90Group/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_rot90aug_vit_small_weight_selection/checkpoints/best-v2.ckpt
  # teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt \
# 
  # 