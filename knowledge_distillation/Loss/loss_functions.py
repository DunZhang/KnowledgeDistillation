import torch
import torch.nn.functional as F


def mse_with_mask(teacher_output, student_output, teacher_input=None, student_input=None):
    mask = teacher_input["attention_mask"]
    mask = mask.to(student_output)
    # * hidden_size
    valid_count = mask.sum() * student_output.size(-1)
    loss = (F.mse_loss(teacher_output, student_output, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss


def attention_mse_with_mask(teacher_output, student_output, teacher_input=None, student_input=None):
    mask = teacher_input["attention_mask"]
    mask = mask.to(student_output).unsqueeze(1).expand(-1, student_output.size(1), -1)  # (bs, num_of_heads, len)
    valid_count = torch.pow(mask.sum(dim=2), 2).sum()
    loss = (F.mse_loss(student_output, teacher_output, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
        2)).sum() / valid_count
    return loss


def mse(teacher_output, student_output, teacher_input=None, student_input=None):
    return F.mse_loss(teacher_output, student_output, reduction='mean')
