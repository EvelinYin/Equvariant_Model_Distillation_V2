import unittest
import torch


import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from components.lifting_convolution_test import test_lifting_convolution_forward, test_lifting_convolution_backward
from components.group_convolution_test import test_group_convolution_forward, test_group_convolution_backward
from components.shared_weight_linear_test import test_shared_weight_linear_forward, test_shared_weight_linear_backward
from components.channel_attention_test import test_channel_attention_forward, test_channel_attention_backward
from components.channel_layer_norm_test import test_channel_layer_norm_forward, test_channel_layer_norm_backward


from models.vit_end_to_end_test import test_vit_forward, test_vit_backward
from models.resnet_end_to_end_test import test_resnet_forward






class TestModels(unittest.TestCase):
    def test_liftingLayer(self):
        # This is forward test
        equ_error = test_lifting_convolution_forward()
        print("Lifting Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        # This is backward test
        equ_error = test_lifting_convolution_backward()
        print("Lifting Conv Layer Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
        
        
    
    def test_groupLayer(self):
        # This is forward test
        equ_error = test_group_convolution_forward()
        print("Group Conv Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        # This is backward test
        equ_error = test_group_convolution_backward()
        print("Group Conv Layer Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
        

    def test_shared_weight_linear(self):
        equ_error = test_shared_weight_linear_forward()
        print("Shared Weight Linear Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10    
        
        equ_error = test_shared_weight_linear_backward()
        print("Shared Weight Linear Layer Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
        
    
    def test_attention_layer(self):
        equ_error = test_channel_attention_forward()
        print("Channel Attention Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10    
        
        equ_error = test_channel_attention_backward()
        print("Channel Attention Layer Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
    
    def test_channel_layer_norm(self):
        equ_error = test_channel_layer_norm_forward()
        print("Channel Layer Norm Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10    
        
        equ_error = test_channel_layer_norm_backward()
        print("Channel Layer Norm Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10

    
    def test_vit_end_to_end(self):
        equ_error = test_vit_forward()
        print("ViT End to End Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        equ_error = test_vit_backward()
        print("ViT End to End Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
    
    
    # def _test_resnet_end_to_end(self):
    #     equ_error = test_resnet_forward()
    #     print("ResNet End to End Equivarinace Error -->", equ_error)
    #     assert equ_error < 1e-10



if __name__ == '__main__':
    unittest.main()
