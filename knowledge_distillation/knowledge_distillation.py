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
            if split_data:
                teacher_batch_data, student_batch_data = split_data(batch_data)
                with torch.no_grad():
                    teacher_output = teacher_model.forward(*(teacher_batch_data[0:3]))
                student_output = student_model.forward(*(student_batch_data[0:3]))
            else:
                with torch.no_grad():
                    teacher_output = teacher_model.forward(*batch_data)
                student_output = student_model.forward(*batch_data)

            # get loss
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
