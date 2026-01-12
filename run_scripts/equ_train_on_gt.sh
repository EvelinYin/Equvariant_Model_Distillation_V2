
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/equ_train_on_gt.yaml \
  logging.wandb_name="192_half_channel_ViT_equ_train_on_gt" \
  student_model.vit_config.embed_dim=192 \
  student_model.vit_config.n_heads=3 \
  student_train.learning_rate=0.002 \
  student_train.epochs=400 \
  student_train.scheduler_warmup_epochs=50 \
  data.batch_size=128 \
  student_train.student_ckpt_path="./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/192_zero_init_uniform_selection.ckpt" \
#   logging.wandb_mode="offline" \


