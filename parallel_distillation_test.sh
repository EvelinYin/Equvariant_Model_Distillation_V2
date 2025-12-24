CUDA_VISIBLE_DEVICES=0 python main.py \
  --config src/exp_configs/vit/parallel_distillation.yaml \
  student_train.student_ckpt_path=/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/student/equ_vit/parallel_distillation/checkpoints/best.ckpt \
  precision=32 \
  data.num_workers=0 \
  test_only=true

# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --config src/exp_configs/vit/parallel_distillation.yaml