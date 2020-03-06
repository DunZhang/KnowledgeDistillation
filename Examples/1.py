import torch

input1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
input2 = torch.FloatTensor([[2, 4, 7], [-4, 5, -6]])
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = 0.5 - 0.5 * cos(input1, input2)
print(output)
