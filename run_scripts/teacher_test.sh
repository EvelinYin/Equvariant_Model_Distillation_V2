
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config src/exp_configs/vit/parallel_distillation.yaml \
  precision=32 \
  data.num_workers=0 \
  train_teacher=true \
  teacher_train.teacher_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt \
  test_only=true \
  logging.wandb_mode="offline" \

