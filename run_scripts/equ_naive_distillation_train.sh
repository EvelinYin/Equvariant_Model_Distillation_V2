
# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/equ_naive_distillation.yaml \
#   logging.wandb_name="192_double_channel_ViT_equ_naive_distillation" \
#   student_model.vit_config.embed_dim=192 \
#   student_model.vit_config.num_heads=3 \
#   student_train.learning_rate=0.0002 \
#   student_train.epochs=500 \
#   student_train.student_ckpt_path="./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/192_zero_init_uniform_selection.ckpt"



CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/equ_naive_distillation.yaml \
  logging.wandb_name="192_half_channel_ViT_equ_naive_distillation" \
  student_model.vit_config.embed_dim=192 \
  student_model.vit_config.n_heads=3 \
  student_train.learning_rate=0.002 \
  student_train.epochs=500 \
  data.batch_size=128 \
  student_train.student_ckpt_path="./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/192_zero_init_uniform_selection.ckpt" \
  teacher_model.pretrained_vit_config.model_name="WinKawaks/vit-small-patch16-224" \
  teacher_train.teacher_ckpt_path="" \



#   logging.wandb_mode="offline" \
