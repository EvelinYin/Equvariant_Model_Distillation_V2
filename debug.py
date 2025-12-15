import sys
# 1. Import the module from where it REALLY is now
# Example: import models.vit.teacher_model as real_module
import src.models.ViT.pretrained_HF as real_module 
import src.models
sys.modules['models'] = src.models

# 2. Tell Python: "When pickle asks for 'models.ViT.teacher_model', give it 'real_module'"
sys.modules['models.ViT.teacher_model'] = real_module

# 3. Sometimes you need to patch the parent package too
import src.models.ViT as real_package
sys.modules['models.ViT'] = real_package

import src.config
sys.modules['config'] = src.config

import torch

# 4. Now load
ckpt_path = "/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best.ckpt"   
breakpoint()
checkpoint = torch.load(ckpt_path, map_location='cpu')

torch.save(checkpoint, "/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt")