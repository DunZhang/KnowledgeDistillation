import torch.nn as nn
import torch
import tqdm


def knowledge_distillation(teacher_model, student_model, train_data, evaluate_data, device,
                           loss_model, optimizer, evaluator, num_epoch, split_data=None):
    teacher_model.eval()
    student_model.train()
    for epoch in range(num_epoch):
        for step, batch_data in enumerate(train_data):
            batch_data = tuple(t.to(device) for t in batch_data)
            # split_data is a function that split train data to teacher and student
            ######################################################################
            ## these codes should be changed according to your task #############
            ######################################################################
            if split_data:
                teacher_batch_data, student_batch_data = split_data(batch_data)
                with torch.no_grad():
                    teacher_output = teacher_model.forward(input_ids=teacher_batch_data[0],
                                                           attention_mask=teacher_batch_data[1],
                                                           token_type_ids=teacher_batch_data[2])
                student_output = student_model.forward(input_ids=student_batch_data[0],
                                                       attention_mask=student_batch_data[1],
                                                       token_type_ids=student_batch_data[2])
            else:
                with torch.no_grad():
                    teacher_output = teacher_model.forward(input_ids=batch_data[0],
                                                           attention_mask=batch_data[1],
                                                           token_type_ids=batch_data[2])
                student_output = student_model.forward(input_ids=batch_data[0],
                                                       attention_mask=batch_data[1],
                                                       token_type_ids=batch_data[2])
            ######################################################################
            # get loss
            # the loss model should be bert loss or pred loss
            loss = loss_model.forward(teacher_output, student_output, batch_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            evaluator.evaluate(teacher_model, student_model, evaluate_data, epoch, step, loss.data)


if __name__ == "__main__":
    pass
    # a = list(range(10000))
    #
    # for i in tqdm(a, desc="Iteration", ascii=True):
    #     print(i)
