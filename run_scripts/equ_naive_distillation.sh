
CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/equ_naive_distillation.yaml \
  logging.wandb_name="double_channel_ViT_equ_naive_distillation" \
  student_train.learning_rate=0.0002 \
  student_train.epochs=150 \
#   logging.wandb_mode="offline" \
