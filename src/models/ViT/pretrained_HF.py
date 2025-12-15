import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

class HFModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # HF models return a generic object (ModelOutput), we just want the logits
        return self.model(x).logits


class PretrainedViT(nn.Module):
    """
    A class-based wrapper that calls your function during __init__.
    """
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=100, device='cuda'):
        super().__init__()
        print(f"Loading {model_name} from Hugging Face...")
    
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        
        
        # Load the HF model
        base_model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Wrap it to normalize the output
        self.model = HFModelWrapper(base_model).to(device)

    def forward(self, x):
        return self.model(x)