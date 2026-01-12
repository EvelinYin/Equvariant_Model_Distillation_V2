
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/canonicalizer.yaml \
  teacher_model.pretrained_vit_config.model_name="WinKawaks/vit-small-patch16-224" \
  train_teacher=true \
  teacher_train.epochs=300 \
  teacher_train.learning_rate=5e-5 \
  teacher_train.scheduler_warmup_epochs=50 \
  teacher_train.weight_decay=5e-2 \
  data.batch_size=128 \
  logging.wandb_mode="online" \


# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/canonicalizer.yaml \
#   train_teacher=true \
#   teacher_train.learning_rate=1e-4 \
#   teacher_train.epochs=300 \
#   teacher_model.canonicalizer_config.use_equ_layers=false \
#   teacher_model.canonicalizer_config.out_channels=2 \
#   logging.wandb_name="canonicalizer_no_equ_layers" \
#   logging.wandb_mode="online" \

