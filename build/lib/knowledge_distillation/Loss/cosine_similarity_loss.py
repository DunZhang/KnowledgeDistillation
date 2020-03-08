import torch
from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, ):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        return 0.5 - 0.5 * torch.cosine_similarity(x1, x2)
