import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

class FaceNeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_views=3,
                 dim_aud=76,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        """ 
        """
        super(FaceNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.dim_aud = dim_aud
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        input_ch_all = input_ch + dim_aud
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W) for i in range(D-1)])

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)])

        # Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        input_views를 이어서 한다.
        """

        input_pts, input_views = torch.split(
            x, [self.input_ch+self.dim_aud, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = h  # self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs