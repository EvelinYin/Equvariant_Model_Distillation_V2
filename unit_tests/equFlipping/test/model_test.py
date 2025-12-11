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
from components.resnet_end_to_end_test import test_resnet_forward






class TestModels(unittest.TestCase):
    def test_liftingLayer(self):
        # This is forward test
        equ_error = test_lifting_convolution_forward()
        print("Lifting Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        # This is backward test
        equ_error = test_lifting_convolution_backward()
        print("Lifting Conv Layer Equivarinace Error (backward) -->", equ_error)
        assert equ_error < 1e-10
        
        
    
    def test_groupLayer(self):
        # This is forward test
        equ_error = test_group_convolution_forward()
        print("Group Conv Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        # This is backward test
        equ_error = test_group_convolution_backward()
        print("Group Conv Layer Equivarinace Error (backward) -->", equ_error)
        assert equ_error < 1e-10
        

    def test_shared_weight_linear(self):
        equ_error = test_shared_weight_linear_forward()
        print("Shared Weight Linear Layer Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10    
        
        equ_error = test_shared_weight_linear_backward()
        print("Shared Weight Linear Layer Equivarinace Error (backward) -->", equ_error)
        assert equ_error < 1e-10
    
    
    def test_resnet_end_to_end(self):
        equ_error = test_resnet_forward()
        print("ResNet End to End Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10



if __name__ == '__main__':
    unittest.main()
