import torch.nn as nn
from .cosine_similarity_loss import CosineSimilarityLoss


class MultiLayerBasedDistillationLoss(nn.Module):
    def __init__(self, distill_config=None, teacher_output_adaptor=None, student_output_adaptor=None):
        super(MultiLayerBasedDistillationLoss, self).__init__()
        self.distill_config = distill_config
        self.teacher_output_adaptor = teacher_output_adaptor
        self.student_output_adaptor = student_output_adaptor
        self.loss_functions = {"mse": nn.MSELoss(), "cross_entropy": nn.CrossEntropyLoss(),
                               "cos": CosineSimilarityLoss()}

    def forward(self, teacher_output, student_output, teacher_input_data, student_input_data):
        teacher_adaptor_output = self.teacher_output_adaptor(teacher_output)
        student_adaptor_output = self.student_output_adaptor(student_output)
        loss = 0
        for distill_info in self.distill_config:
            if distill_info["type"] == "ts_distill":
                tmp_teacher_output = teacher_adaptor_output[distill_info["teacher_layer_name"]][
                    distill_info["teacher_layer_output_name"]]
                tmp_student_output = student_adaptor_output[distill_info["student_layer_name"]][
                    distill_info["student_layer_output_name"]]
                tmp_loss = distill_info["loss"]["loss_function"](tmp_teacher_output, tmp_student_output,
                                                                 teacher_input_data, student_input_data,
                                                                 ** distill_info["loss"]["args"])
                tmp_loss *= distill_info["weight"]
            elif distill_info["type"] == "hard_distill":
                tmp_student_output = student_adaptor_output[distill_info["student_layer_name"]][
                    distill_info["student_layer_output_name"]]
                tmp_loss = distill_info["loss"]["loss_function"](tmp_student_output, student_input_data,
                                                                 **distill_info["loss"]["args"])
                tmp_loss *= distill_info["weight"]

            loss += tmp_loss

        # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)  # 小于-100的部分变成0
        return loss
