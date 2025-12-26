
# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml


CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
  --config src/exp_configs/vit/parallel_distillation.yaml \
  student_train.learning_rate=0.001 \
  student_train.epochs=500 \
  student_model.vit_config.embed_dim=384 \
  parallel_layer_distillation.learnable_projection=true \
  logging.wandb_name="half_channel_learnable_projection_ViT" \
  student_train.student_ckpt_path="/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init_v2.ckpt"


# CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml \
#   logging.wandb_name="random_double_channel_init_ViT" \
#   student_train.student_ckpt_path='' \
#   student_train.learning_rate=0.0001 \
#   student_train.epochs=300 \




# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml