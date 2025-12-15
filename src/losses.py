import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_relative_error(teacher_tensor, student_tensor, norm_type=2):
    """Compute relative error percentage between two image arrays."""
    if teacher_tensor.shape != student_tensor.shape:
        breakpoint()
        raise ValueError("Must have the same dimensions.")
    
    from numpy.linalg import norm
    student_tensor = student_tensor.cpu()
    teacher_tensor = teacher_tensor.cpu()
    diff = student_tensor - teacher_tensor
    numerator = norm(diff.ravel(), ord=norm_type)
    denominator = norm(teacher_tensor.ravel(), ord=norm_type)
    
    if denominator == 0:
        return float('inf')

    return numerator / denominator



def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    Compute distillation loss combining soft targets and hard labels
    
    Args:
        student_logits: Student model output logits
        teacher_logits: Teacher model output logits
        
    Returns:
        Total loss
    """
    T = temperature
    # T = self.train_config.temperature
    # alpha = self.train_config.alpha
    
    # Hard label loss
    # hard_loss = self.criterion(student_logits, labels)
    
    # Soft label loss (KL divergence with temperature)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    return soft_loss
    
    # Combined loss
    # total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    # return total_loss, hard_loss, soft_loss
