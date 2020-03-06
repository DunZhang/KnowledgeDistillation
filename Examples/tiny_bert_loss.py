import torch.nn as nn
import torch


class TinyBERTLoss(nn.Module):
    def __init__(self, compute_cls_loss=False):
        super(TinyBERTLoss, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.compute_cls_loss = compute_cls_loss

    def forward(self, teacher_output, student_output, batch_data):
        teacher_pooler_output, teacher_atts, teacher_reps = teacher_output
        student_pooler_output, student_atts, student_reps = student_output

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)  #
        # compute attention loss
        att_loss = 0.
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]  # 为了和学生模型一一对应
                            for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)  # 小于-100的部分变成0
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)  # 小于-100的部分变成0
            tmp_loss = self.loss_mse(student_att, teacher_att)
            att_loss += tmp_loss
        # compute hidden_loss and embedding_loss
        rep_loss = 0.
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps
        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            tmp_loss = self.loss_mse(student_rep, teacher_rep)
            rep_loss += tmp_loss
        # compute cls loss
        cls_loss = 0.
        if self.compute_cls_loss:
            cls_loss = self.loss_mse(student_pooler_output, teacher_pooler_output)
        loss = rep_loss + att_loss + cls_loss

        return loss
