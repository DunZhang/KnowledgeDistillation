# This is an simple example to distill bert by using the loss of tiny bert


from knowledge_distillation import knowledge_distillation
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from transformers import BertModel, BertConfig
from knowledge_distillation.Optimizer import BertAdam
from knowledge_distillation.Loss import BERTLoss
from knowledge_distillation.Evaluator import TinyBERTEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger()

if __name__ == "__main__":
    # some parameters
    train_batch_size = 36
    num_train_epochs = 20
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We just generate some random data
    input_ids = torch.LongTensor(np.random.randint(0, 10000, (1000, 64)))
    attention_mask = torch.LongTensor(np.ones((10000, 64)))
    token_type_ids = torch.LongTensor(np.zeros((10000, 64)))

    train_data = TensorDataset(input_ids, attention_mask, token_type_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    # load construct models
    # for teacher model, we use pretrained model
    # teacher_model = TinyBERT.from_pretrained("bert-base-uncased",output_hidden_states=True,output_attentions=True)
    teacher_model = BertModel.from_pretrained(r"G:\Data\BERTModel\torch\uncased_L-12_H-768_A-12",
                                              output_hidden_states=True, output_attentions=True)
    teacher_model.to(device)
    # for student model, we use three layers of bert
    bert_config = BertConfig(num_hidden_layers=3, output_hidden_states=True, output_attentions=True)
    student_model = BertModel(bert_config)
    student_model.to(device)
    # prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         schedule="none",
                         lr=learning_rate,
                         warmup=0.1,
                         t_total=int(len(input_ids) / train_batch_size) * num_train_epochs)

    # loss model
    loss_model = BERTLoss(compute_cls_loss=False)
    # evalator
    evaluator = TinyBERTEvaluator(save_dir=None, save_step=None)

    knowledge_distillation(teacher_model=teacher_model, student_model=student_model, train_data=train_dataloader,
                           evaluate_data=None, device=device, loss_model=loss_model, optimizer=optimizer,
                           evaluator=evaluator, num_epoch=num_train_epochs, split_data=None)
