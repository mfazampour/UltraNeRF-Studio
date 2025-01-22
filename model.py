from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from camera import Lie, Pose


class NeRF(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=6, skips=[4]
    ):
        """ """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)

        return outputs

class BARF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=6, skips=[4], L=0, c2f=None):
        """ """
        super(BARF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.L = L
        self.c2f = c2f

        if self.L > 0:
            input_ch = input_ch + 2 * input_ch * L

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.output_linear = nn.Linear(W, output_ch)

        self.progress = nn.Parameter(torch.tensor(0.0))

    def positional_encoding_bundle(self, inp, L, c2f=None):  # [B,...,N]

        input_enc = self.positional_encoding(inp, L=L)  # [B,...,2NL]

        # coarse-to-fine: smoothly mask positional encoding for BARF

        if c2f is not None:
            # set weights for different frequency bands
            start, end = c2f
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=inp.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)

        return input_enc

    def positional_encoding(self, x, L):  # [B,...,N]
        shape = x.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=x.device) * np.pi  # [L]
        spectrum = x.unsqueeze(-1) * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]

        return input_enc

    def forward(self, x):

        # apply positional encoding
        if self.L > 0:
            if self.c2f is not None:
                points_enc = self.positional_encoding_bundle(x, L=self.L, c2f=self.c2f)
            else:
                points_enc = self.positional_encoding(x, L=self.L)
            input_pts = torch.cat([x, points_enc], -1)  # [B,...,N+2NL]

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)

        return outputs


class PoseRefine(nn.Module):
    def __init__(self, poses=Optional[torch.Tensor], mode=Optional[str]):
        super(PoseRefine, self).__init__()
        self.poses = poses
        self.mode = mode
        self.lie = Lie()
        self.pose = Pose()

        if self.poses is not None:
            self.se3_refine = nn.Embedding(poses.shape[0], 6)
            torch.nn.init.zeros_(self.se3_refine.weight)

    def set_poses(self, poses: torch.Tensor):
        self.poses = poses

    def get_pose(self, idx: torch.Tensor):

        if self.poses is None:
            raise ValueError("No poses available")

        if self.mode == "train":
            se3_refine = self.se3_refine.weight[idx]
            pose = self.poses[idx].to(se3_refine.device)
            pose_refine = self.lie.se3_to_SE3(se3_refine)
            pose = self.pose.compose([pose_refine, pose])

            return pose

        else:
            raise NotImplementedError
