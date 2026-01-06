
# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/canonicalizer.yaml \
#   train_teacher=true \
#   teacher_train.learning_rate=1e-4 \
#   teacher_train.epochs=300 \
#   logging.wandb_mode="online" \


CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/canonicalizer.yaml \
  train_teacher=true \
  teacher_train.learning_rate=1e-4 \
  teacher_train.epochs=300 \
  teacher_model.canonicalizer_config.use_equ_layers=false \
  teacher_model.canonicalizer_config.out_channels=2 \
  logging.wandb_name="canonicalizer_no_equ_layers" \
  logging.wandb_mode="online" \

