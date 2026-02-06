
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/canonicalizer.yaml \
#   train_teacher=true \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/canonicalizer/checkpoints/best-v1.ckpt \
#   precision=64 \
#   data.num_workers=0 \
#   test_only=true \
#   logging.wandb_mode="online" \


# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/canonicalizer.yaml \
#   train_teacher=true \
#   teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/canonicalizer_no_equ_layers/checkpoints/best.ckpt \
#   precision=64 \
#   data.num_workers=0 \
#   teacher_model.canonicalizer_config.use_equ_layers=false \
#   teacher_model.canonicalizer_config.out_channels=2 \
#   test_only=true \
#   logging.wandb_mode="online" \


CUDA_VISIBLE_DEVICES=2 python main.py \
  --config src/exp_configs/vit/canonicalizer.yaml \
  train_teacher=true \
  teacher_model.pretrained_vit_config.model_name="WinKawaks/vit-small-patch16-224" \
  teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/Rot45Group/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/rot45_canonicalizer_noaug_equ_layers/checkpoints/best.ckpt \
  teacher_train.group="Rot45Group" \
  precision=64 \
  data.num_workers=8 \
  data.batch_size=64 \
  teacher_model.canonicalizer_config.use_equ_layers=true \
  test_only=true \
  logging.wandb_mode="online" \
  logging.wandb_name="test_rot45_canonicalizer_noaug_equ_layers" \






# 