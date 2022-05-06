import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

class AudioAttnNet(nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttnNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # [2, 16, 76] > [2, 76, 16] > [1, 76, 32]
        y = x[..., :self.dim_aud].permute(0, 2, 1)#.unsqueeze(0)  # 2 x subspace_dim x seq_len
        #y = x[..., :self.dim_aud].permute(0, 2, 1)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y) # [2,1,8]
        y = self.attentionNet(y.view(-1, self.seq_len)).view(-1, self.seq_len, 1)
        # print(y.view(-1).data)
        return torch.sum(y*x, dim=1)