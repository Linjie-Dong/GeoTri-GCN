import torch
from torch import nn


class TriAxisStreamFusion(nn.Module):
    """Fuse input streams with stream-time, stream-joint and local time-joint cues.

    All input branches keep the same (C, T, V) resolution.  The learned softmax
    therefore selects between physical input forms (joint, velocity and bone),
    rather than between feature-pyramid resolutions as in image ASFF modules.
    """

    def __init__(self, num_streams, A, temporal_kernel=3):
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError('fusion temporal kernel must be odd')

        self.num_streams = num_streams
        padding = (temporal_kernel - 1) // 2

        # Local time-joint evidence. Average and max descriptors retain
        # complementary responses while avoiding a channel-dependent head.
        self.local_proj = nn.Conv2d(num_streams * 2, num_streams, 1, bias=True)

        # Stream-time interaction, shared across joints.
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(
                num_streams, num_streams, (temporal_kernel, 1),
                padding=(padding, 0), groups=num_streams, bias=False,
            ),
            nn.Conv2d(num_streams, num_streams, 1, bias=True),
        )

        # Stream-joint interaction. Only self and one-hop graph relations are
        # used, so the operation respects skeleton topology rather than the
        # arbitrary numeric order of joints.
        graph = A[:min(2, A.size(0))].sum(0).float()
        graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer('graph', graph)
        self.joint_proj = nn.Conv2d(num_streams, num_streams, 1, bias=True)

        # Zero strength makes the initial output equal to channel concatenation.
        self.strength = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        if len(features) != self.num_streams:
            raise ValueError(
                'expected {} input streams, got {}'.format(
                    self.num_streams, len(features)
                )
            )

        x = torch.stack(features, dim=1)  # N, S, C, T, V

        avg_desc = x.mean(2)
        max_desc = x.amax(2)
        local_score = self.local_proj(torch.cat([avg_desc, max_desc], dim=1))

        temporal_desc = x.mean(dim=(2, 4)).unsqueeze(-1)  # N, S, T, 1
        temporal_score = self.temporal_proj(temporal_desc)

        joint_desc = x.mean(dim=(2, 3)).unsqueeze(2)  # N, S, 1, V
        joint_desc = torch.einsum('nsiv,vw->nsiw', joint_desc, self.graph)
        joint_score = self.joint_proj(joint_desc)

        score = local_score + temporal_score + joint_score
        weight = torch.softmax(score, dim=1)

        alpha = torch.tanh(self.strength)
        scale = 1.0 + alpha * (self.num_streams * weight - 1.0)
        x = x * scale.unsqueeze(2)

        return torch.cat([x[:, i] for i in range(self.num_streams)], dim=1)


class SelectiveMultiScaleTemporal(nn.Module):
    """A lightweight skeleton-specific adaptation of selective-kernel attention."""

    def __init__(self, channel, kernels=(3, 5, 7), reduction=4):
        super().__init__()
        if not kernels or any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError('temporal kernels must be a non-empty list of odd values')

        hidden = max(channel // reduction, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channel, channel, (kernel, 1),
                    padding=((kernel - 1) // 2, 0),
                    groups=channel, bias=False,
                ),
                nn.BatchNorm2d(channel),
            )
            for kernel in kernels
        ])
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.selectors = nn.ModuleList([
            nn.Conv2d(hidden, channel, 1, bias=True) for _ in kernels
        ])
        self.project = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.strength = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        context = self.reduce(branch_features.sum(1))
        score = torch.stack([selector(context) for selector in self.selectors], dim=1)
        weight = torch.softmax(score, dim=1)
        fused = (branch_features * weight).sum(1)
        return x + torch.tanh(self.strength) * self.project(fused)
