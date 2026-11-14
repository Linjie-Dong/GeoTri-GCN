import torch
from torch import nn


class _GradientReverse(torch.autograd.Function):
    """Pass features unchanged and reverse only their backward gradient."""

    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x, coefficient):
    return _GradientReverse.apply(x, coefficient)


class PersonAdversarialHead(nn.Module):
    """Linear subject discriminator used only while training."""

    def __init__(self, input_channel, num_person_class):
        super(PersonAdversarialHead, self).__init__()
        if num_person_class < 2:
            raise ValueError('num_person_class must be at least 2')
        self.num_person_class = int(num_person_class)
        self.fc = nn.Linear(input_channel, self.num_person_class)

    def forward(self, feature, coefficient):
        if feature.dim() != 2:
            raise ValueError('PersonAdversarialHead expects a [N, C] feature tensor')
        return self.fc(gradient_reverse(feature, coefficient))
