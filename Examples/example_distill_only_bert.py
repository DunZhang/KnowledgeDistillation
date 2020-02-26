# This is an simple example to distill bert by using the loss of tiny bert


from knowledge_distillation import knowledge_distillation
import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from knowledge_distillation.Model import TinyBERT
from transformers import BertTokenizer
from knowledge_distillation.DataProcessor import InputExample, InputFeatures, TripletProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger()


def smart_batch_(x):
    """ 动态batch """
    # print([i[-1] for i in x])
    seq_lengths = torch.tensor([i[-1] for i in x], dtype=torch.long)
    label_ids = torch.tensor([i[-2] for i in x], dtype=torch.long)
    max_seq_len = torch.max(seq_lengths)
    input_ids = torch.cat([i[0][0:max_seq_len].unsqueeze(0) for i in x], dim=0).long()
    input_mask = torch.cat([i[1][0:max_seq_len].unsqueeze(0) for i in x], dim=0).long()
    segment_ids = torch.cat([i[2][0:max_seq_len].unsqueeze(0) for i in x], dim=0).long()
    return input_ids, input_mask, segment_ids, label_ids, seq_lengths

def smart_batch(x):
    return  smart_batch_(x[0:5])+smart_batch_(x[5:])

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def features_to_tensor(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths

names = ['db_zhongyingrenshou_20190409#573', 'db_youzu_20190409#101', 'robot3_meidi_20190528#101',
         'db_youzu_20190409#109', 'db_fuweike_20190409#101', 'db_beijingcanlian_20190409#101',
         'db_zhubajie_20190409#19203', 'db_ppmoney_20190409#101', 'robot3_jinli_20180131#101', 'robot3_xishanju#101',
         'robot3_debang_20190109#101', 'robot3_huatai_20180630#101', 'robot3_meidi0830#101',
         'robot3_lanling_20180807#145', 'db_qianjiwang_20190409#125', 'db_zhubajie_20190409#19205',
         'robot3_daru_20180111#34', 'robot4_huazhu_20180605#102', 'robot3_fanli_20180430#101',
         'robot3_dunhuang_20180404#101', 'db_beiligong_20190409#339', 'db_guanglianda_20190409#101',
         'db_shanghaitushu_20190409#17169', 'db_debang_duinei_20190409#101', 'robot3_dianli_0428#101',
         'db_boxijiadian_20190409#105', 'db_zhaobiaowang_20190409#101', 'robot4_haikang_20180606#83',
         'db_saikesi_20190409#1037', 'robot4_wuxianji_20180615#34', 'robot3_fangxin_20190125#101',
         'db_guotairenshou_20190409#101', 'robot3_ziru_20180828#101'][0:2]
if __name__ == "__main__":
    train_data_dir = r"E:\云问\数据\task6triple\triple\train"
    train_file_names = [i + ".txt" for i in names]
    do_lower_case = True
    max_seq_length = 128
    train_batch_size = 36
    gradient_accumulation_steps = 1
    num_train_epochs = 10
    student_model_dir = r"G:\Data\rbt3"
    teacher_model_dir = r"G:\Codes\PythonProj\SBERT\output\SimBERTCLS2e6\0_BERT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TripletProcessor(train_data_dir)
    output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    student_tokenizer = BertTokenizer.from_pretrained(student_model_dir, do_lower_case=do_lower_case)
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_dir, do_lower_case=do_lower_case)

    # 准备训练数据

    train_examples = processor.get_examples(train_file_names)
    train_batch_size = train_batch_size // gradient_accumulation_steps

    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

    student_train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, student_tokenizer,
                                                          output_mode)
    teacher_train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, teacher_tokenizer,
                                                          output_mode)
    # train_data: TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids, all_seq_lengths)
    student_train_data = features_to_tensor(output_mode,student_train_features)
    teacher_train_data = features_to_tensor(output_mode,teacher_train_features)

    train_data = TensorDataset(*teacher_train_data,*student_train_data)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size,
                                  collate_fn=smart_batch)
    for batch_data in train_dataloader:
        print(len(batch_data),batch_data[0].shape,batch_data[5].shape)