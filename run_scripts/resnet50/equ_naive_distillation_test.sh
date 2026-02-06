
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config src/exp_configs/vit/equ_naive_distillation.yaml \
  logging.wandb_name="double_channel_ViT_equ_naive_distillation" \
  student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/equ_naive_distillationdouble_channel_ViT_equ_naive_distillation/checkpoints/best-v4.ckpt \
  test_only=true \
  precision=32 \
  data.num_workers=0 \
  logging.wandb_mode="offline" \
