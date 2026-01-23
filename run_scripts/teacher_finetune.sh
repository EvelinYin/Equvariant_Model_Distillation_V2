export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --config src/exp_configs/vit/teacher_finetune.yaml \
#   data.num_workers=0 \
#   train_teacher=true \
#   teacher_model.pretrained_vit_config.model_name="WinKawaks/vit-tiny-patch16-224" \
#   teacher_train.teacher_ckpt_path=./outputs/CIFAR100/pretrained_ViT/student/initialization/vit_tiny_teacher/192_zero_init_uniform_selection.ckpt \
#   teacher_train.epochs=300 \
#   teacher_train.learning_rate=0.002 \
#   teacher_train.scheduler_warmup_epochs=50 \
#   data.batch_size=128 \
#   logging.wandb_name="nonequ_teacher_vit_tiny_weight_selection" \
#   logging.wandb_mode="online" \


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --config src/exp_configs/vit/teacher_finetune.yaml \
  train_teacher=true \
  teacher_model.pretrained_vit_config.model_name="WinKawaks/vit-small-patch16-224" \
  teacher_train.epochs=500 \
  teacher_train.learning_rate=2e-5 \
  teacher_train.scheduler_warmup_epochs=50 \
  teacher_train.weight_decay=5e-2 \
  teacher_train.group="Rot45Group" \
  data.batch_size=256 \
  logging.wandb_name="teacher_rot45aug_vit_small_weight_selection" \
  logging.wandb_mode="online" \