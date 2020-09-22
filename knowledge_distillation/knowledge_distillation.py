import torch


class KnowledgeDistiller():
    def __init__(self, teacher_model, student_model, train_dataloader, dev_dataloader, device, loss_model, optimizer,
                 evaluator, num_epoch, train_data_adaptor, dev_data_adaptor):
        # send model to device
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.device = device
        self.loss_model = loss_model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.num_epoch = num_epoch
        self.train_data_adaptor = train_data_adaptor
        self.dev_data_adaptor = dev_data_adaptor

    def distillate(self):
        # not train teacher model
        self.teacher_model.eval()
        # train student model
        self.student_model.train()
        for epoch in range(self.num_epoch):
            for step, batch_data in enumerate(self.train_dataloader):
                # get input data for teacher model and student model
                teacher_batch_data, student_batch_data = self.train_data_adaptor(self.device, batch_data)
                # get teacher output, not compute gradient
                with torch.no_grad():
                    if isinstance(teacher_batch_data, dict):
                        teacher_output = self.teacher_model.forward(**teacher_batch_data)
                    else:
                        teacher_output = self.teacher_model.forward(*teacher_batch_data)
                # get student output
                if isinstance(student_batch_data, dict):
                    student_output = self.student_model.forward(**student_batch_data)
                else:
                    student_output = self.student_model.forward(*student_batch_data)
                # get loss
                loss = self.loss_model.forward(teacher_output, student_output, teacher_batch_data, student_batch_data)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # evaluate and save model
                self.evaluator.evaluate(self.teacher_model, self.student_model, self.dev_dataloader,
                                        self.dev_data_adaptor, epoch, step, loss)
