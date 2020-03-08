KnowledgeDistillation
======================


Introduction
------------

What is knowledge distillation?
:::::::::::::::::::::::::::::::::::::::::
**Knowledge Distillation** is model compression method in which a small model is trained 
to mimic a pre-trained, larger model (or ensemble of models). Recently, many models have achieved SOTA performance.
However, their billions of parameters make it computationally expensive and inefficient considering both memory 
consumption and high latency. Hence, it is necessary to get a small model from a large model by using knowledge 
distillation.

KnowledgeDistillation's training setting is sometimes referred to as "teacher-student", 
where the large model is the teacher and the small model is the student.
The method was first proposed by `Bucila <https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf>`_
and generalized by `Hinton <https://arxiv.org/abs/1503.02531>`_.

Introduction of KnowledgeDistillation Package
:::::::::::::::::::::::::::::::::::::::::::::::
**KnowledgeDistillation**  is a general knowledge distillation framework. You can distill your own model
by using this toolkit. Our framework is highly abstract and you can achieve many distillation methods by using this framework.
Besides, we also provide a distillation of MultiLayerBasedModel considering many models are multi layers.

Usage
--------

To use the package, you should define these objects:

* **Teacher Model** (large model, trained)
* **Student Model** (small model, untrained)
* **Data loader**, a generator or iterator to get training data or dev data. For example, torch.utils.data.DataLoader
* **Train data adaptor**, a function that turn batch_data (from train_dataloader) to the inputs of teacher_model and student_model
* **Distill config**, a list-object, each item indicates how to calculate loss. It also defines which output of which layer to calculate loss.
* **Output adaptor**, a function that turn your model's output to dict-object output which meet distiller's requirements
* **Evaluator**, a class with evaluate function, it define when and how to save your student model


Installation
---------------
Requirements
::::::::::::::::::
- Python >= 3.6
- PyTorch >= 1.1.0
- NumPy
- Transformers >= 2.0 (optional, used by some examples)

Install from PyPI
::::::::::::::::::

**KnowledgeDistillation**  is currently available on the PyPi's repository and you can
install it via pip::

 pip install -U KnowledgeDistillation

Install from the Github
::::::::::::::::::::::::::::::
If you prefer, you can clone it and run the setup.py file. Use the following
command to get a copy from GitHub::

 git clone https://github.com/DunZhang/KnowledgeDistillation.git

TODO List
-------------
* Add multi teacher model distiller
* Use input mask when computing loss
* Support custom loss functions

A simple example
----------------
A simple example::

    import torch
    import logging
    import numpy as np
    from transformers import BertModel, BertConfig
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset

    from knowledge_distillation import KnowledgeDistiller, MultiLayerBasedDistillationLoss
    from knowledge_distillation import MultiLayerBasedDistillationEvaluator

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Some global variables
    train_batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 2e-5
    num_epoch = 10
    # Teacher Model
    bert_config = BertConfig(num_hidden_layers=12, output_hidden_states=True, output_attentions=True)
    teacher_model = BertModel(bert_config)
    # Student Model
    bert_config = BertConfig(num_hidden_layers=3, output_hidden_states=True, output_attentions=True)
    student_model = BertModel(bert_config)

    ### Train data loader
    input_ids = torch.LongTensor(np.random.randint(100, 1000, (100000, 64)))
    attention_mask = torch.LongTensor(np.ones((100000, 64)))
    token_type_ids = torch.LongTensor(np.zeros((100000, 64)))
    train_data = TensorDataset(input_ids, attention_mask, token_type_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)


    ### Train data adaptor
    ### It is a function that turn batch_data (from train_dataloader) to the inputs of teacher_model and student_model
    ### You can define your own train_data_adaptor. Remember the input must be device and batch_data.
    ###  The output is either dict or tuple, but must consistent with you model's input
    def train_data_adaptor(device, batch_data):
        batch_data = tuple(t.to(device) for t in batch_data)
        batch_data_dict = {"input_ids": batch_data[0],
                           "attention_mask": batch_data[1],
                           "token_type_ids": batch_data[2], }
        # In this case, the teacher and student use the same input
        return batch_data_dict, batch_data_dict


    ### The loss model is the key for this generation.
    ### We have already provided a general loss model for distilling multi bert layer
    ### In most cases, you can directly use this model.
    #### First, we should define a distill_config which indicates how to compute ths loss between teacher and student.
    #### distill_config is a list-object, each item indicates how to calculate loss.
    #### It also defines which output of which layer to calculate loss.
    #### type "ts_distill" means that we compute loss between teacher and student
    #### type "hard_distill" means that we compute loss between student output and ground truth
    #### loss_function can be mse, cross_entropy or cos. Args is extra parameters in this loss_function
    #### loss_function(x,y,**args)
    distill_config = [
        {"type": "ts_distill",
         "teacher_layer_name": "embedding_layer", "teacher_layer_output_name": "embedding",
         "student_layer_name": "embedding_layer", "student_layer_output_name": "embedding",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer4", "teacher_layer_output_name": "hidden_states",
         "student_layer_name": "bert_layer1", "student_layer_output_name": "hidden_states",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer4", "teacher_layer_output_name": "attention",
         "student_layer_name": "bert_layer1", "student_layer_output_name": "attention",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer8", "teacher_layer_output_name": "hidden_states",
         "student_layer_name": "bert_layer2", "student_layer_output_name": "hidden_states",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer8", "teacher_layer_output_name": "attention",
         "student_layer_name": "bert_layer2", "student_layer_output_name": "attention",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer12", "teacher_layer_output_name": "hidden_states",
         "student_layer_name": "bert_layer3", "student_layer_output_name": "hidden_states",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
        {"type": "ts_distill",
         "teacher_layer_name": "bert_layer12", "teacher_layer_output_name": "attention",
         "student_layer_name": "bert_layer3", "student_layer_output_name": "attention",
         "loss": {"loss_function": "mse", "args": {}}, "weight": 1.0
         },
    ]

    ### teacher_output_adaptor and student_output_adaptor
    ### In most cases, model's output is tuple-object, However, in our package, we need the output is dict-object,
    ### like: { "layer_name":{"output_name":value} .... }
    ### Hence, the output adaptor is to turn your model's output to dict-object output
    ### In my case, teacher and student can use one adaptor
    def output_adaptor(model_output):
        last_hidden_state, pooler_output, hidden_states, attentions = model_output
        output = {"embedding_layer": {"embedding": hidden_states[0]}}
        for idx in range(len(attentions)):
            output["bert_layer" + str(idx + 1)] = {"hidden_states": hidden_states[idx + 1],
                                                   "attention": attentions[idx]}
        return output


    # loss_model
    loss_model = MultiLayerBasedDistillationLoss(distill_config=distill_config,
                                                 teacher_output_adaptor=output_adaptor,
                                                 student_output_adaptor=output_adaptor)
    # optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.Adam(params=optimizer_grouped_parameters, lr=learning_rate)
    # evaluator
    evaluator = MultiLayerBasedDistillationEvaluator(save_dir=None, save_step=None, print_loss_step=20)
    # Get a KnowledgeDistiller
    distiller = KnowledgeDistiller(teacher_model=teacher_model, student_model=student_model,
                                   train_dataloader=train_dataloader, dev_dataloader=None,
                                   train_data_adaptor=train_data_adaptor, dev_data_adaptor=None,
                                   device=device, loss_model=loss_model, optimizer=optimizer,
                                   evaluator=evaluator, num_epoch=num_epoch)
    # start distillate
    distiller.distillate()


