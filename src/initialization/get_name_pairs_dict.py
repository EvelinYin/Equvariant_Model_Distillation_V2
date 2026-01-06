import torch
import re
from collections import OrderedDict

def create_layer_mapping(teacher_model, student_model):
    """
    Create a mapping dictionary from teacher model layer names to student model layer names.
    
    Args:
        teacher_model: Teacher ViT model
        student_model: Student ViT model with same structure but different layer names
    
    Returns:
        dict: Mapping from teacher layer names to student layer names
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    teacher_keys = list(teacher_state.keys())
    student_keys = list(student_state.keys())
    
    # Check if models have same number of parameters
    if len(teacher_keys) != len(student_keys):
        raise ValueError(f"Models have different number of parameters: "
                        f"Teacher={len(teacher_keys)}, Student={len(student_keys)}")
    
    # Create mapping by matching shapes
    mapping = OrderedDict()
    used_student_keys = set()
    
    for teacher_key in teacher_keys:
        teacher_shape = teacher_state[teacher_key].shape
        
        # Find matching student key with same shape
        matched = False
        for student_key in student_keys:
            if student_key in used_student_keys:
                continue
                
            student_shape = student_state[student_key].shape
            
            if teacher_shape == student_shape:
                mapping[teacher_key] = student_key
                used_student_keys.add(student_key)
                matched = True
                break
        
        if not matched:
            print(f"Warning: No match found for teacher key: {teacher_key} with shape {teacher_shape}")
    
    return mapping


def transfer_weights(teacher_model, student_model, mapping=None):
    """
    Transfer weights from teacher to student model using the mapping.
    
    Args:
        teacher_model: Source model
        student_model: Target model
        mapping: Dictionary mapping teacher keys to student keys (optional)
    
    Returns:
        student_model: Student model with transferred weights
    """
    if mapping is None:
        mapping = create_layer_mapping(teacher_model, student_model)
    
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    for teacher_key, student_key in mapping.items():
        student_state[student_key] = teacher_state[teacher_key].clone()
    
    student_model.load_state_dict(student_state)
    return student_model


def save_mapping(mapping, filepath='layer_mapping.txt'):
    """Save the mapping to a text file."""
    with open(filepath, 'w') as f:
        for teacher_key, student_key in mapping.items():
            f.write(f"{teacher_key} -> {student_key}\n")
    print(f"Mapping saved to {filepath}")


def load_mapping(filepath='layer_mapping.txt'):
    """Load the mapping from a text file."""
    mapping = OrderedDict()
    with open(filepath, 'r') as f:
        for line in f:
            if '->' in line:
                teacher_key, student_key = line.strip().split(' -> ')
                mapping[teacher_key] = student_key
    return mapping


# Example usage
if __name__ == "__main__":
    # Example: Load your models
    # teacher_model = YourTeacherViT(...)
    # student_model = YourStudentViT(...)
    
    # Or load from checkpoints
    # teacher_model = torch.load('teacher.pth')
    # student_model = torch.load('student.pth')
    
    # Create the mapping
    # mapping = create_layer_mapping(teacher_model, student_model)
    
    # Print the mapping
    # print("Layer name mapping:")
    # for teacher_key, student_key in mapping.items():
    #     print(f"{teacher_key:50s} -> {student_key}")
    
    # Save mapping to file
    # save_mapping(mapping, 'vit_layer_mapping.txt')
    
    # Transfer weights
    # student_model = transfer_weights(teacher_model, student_model, mapping)
    
    # Save student model with transferred weights
    # torch.save(student_model.state_dict(), 'student_with_teacher_weights.pth')
    
    print("Script loaded. Uncomment the example usage section to run with your models.")