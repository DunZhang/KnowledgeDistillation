from transformers import BertModel
import torch.nn as nn
import logging
import os


class TinyBERT(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super(TinyBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True,
                                              output_attentions=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        last_hidden_state, pooler_output, hidden_states, attentions = self.bert(input_ids, attention_mask,
                                                                                token_type_ids)
        return pooler_output, attentions, hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        return cls(pretrained_model_name_or_path)

    def save(self, output_path):
        # 保存bert
        self.bert.save_pretrained(output_path)
        # TODO 保存其他的模型


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    #####################################################################################################################
    model_path = r"G:\Data\rbt3"
    model_path = r"G:\Data\BERTModel\torch\chinese_L-12_H-768_A-12"
    model_path = r"G:\Codes\PythonProj\SBERT\output\RobertMean\0_BERT"
    # model_path = r"G:\Data\simbert_torch"
    model = TinyBERT.from_pretrained(model_path)
