import unittest
import torch


import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from models.vit_end_to_end_test import test_vit_forward, test_vit_backward
from models.canonicalization_test import test_canonicalization_forward, test_canonicalization_backward
from models.resnet_end_to_end_test import test_resnet_forward






class TestModels(unittest.TestCase):
    def test_vit_end_to_end(self):
        # equ_error = test_vit_forward()
        # print("ViT End to End Equivarinace Error -->", equ_error)
        # assert equ_error < 1e-10
        
        equ_error = test_vit_backward()
        print("ViT End to End Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
    
    def _test_canonicalization_forward(self):
        equ_error = test_canonicalization_forward()
        print("Canonicalization Network Equivarinace Error -->", equ_error)
        assert equ_error < 1e-10
        
        equ_error = test_canonicalization_backward()
        print("Canonicalization Network Equivarinace Error (backward) -->", equ_error, "\n")
        assert equ_error < 1e-10
    
    
    # def _test_resnet_end_to_end(self):
    #     equ_error = test_resnet_forward()
    #     print("ResNet End to End Equivarinace Error -->", equ_error)
    #     assert equ_error < 1e-10



if __name__ == '__main__':
    unittest.main()
