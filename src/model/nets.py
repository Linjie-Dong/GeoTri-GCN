import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .fusions import SelectiveMultiScaleTemporal, TriAxisStreamFusion
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer
from .person_invariance import PersonAdversarialHead


class EfficientGCN(nn.Module):
    def __init__(
        self, data_shape, block_args, fusion_stage, stem_channel,
        fusion_type='concat', fusion_temporal_kernel=3,
        temporal_refine=False, temporal_refine_kernels=(3, 5, 7),
        temporal_refine_reduction=4,
        person_adversarial=False, num_person_class=0,
        grl_max=0.0, grl_gamma=10.0, person_loss_weight=1.0,
        **kwargs
    ):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        fusion_channel = num_input * last_channel
        if fusion_type == 'concat':
            self.stream_fusion = None
        elif fusion_type == 'tri_axis':
            self.stream_fusion = TriAxisStreamFusion(
                num_streams=num_input,
                A=kwargs['A'],
                temporal_kernel=fusion_temporal_kernel,
            )
        else:
            raise ValueError('Unsupported fusion type: {}'.format(fusion_type))

        self.temporal_refine = SelectiveMultiScaleTemporal(
            channel=fusion_channel,
            kernels=tuple(temporal_refine_kernels),
            reduction=temporal_refine_reduction,
        ) if temporal_refine else None

        self.main_stream = EfficientGCN_Blocks(
            init_channel = fusion_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # Initialize the action network before the optional identity head.
        init_param(self.modules())
        self.person_head = None
        if person_adversarial:
            # Do not let auxiliary-head initialization change the random
            # stream later used by DataLoader shuffling and action dropout.
            with torch.random.fork_rng(devices=[]):
                self.person_head = PersonAdversarialHead(
                    last_channel, num_person_class
                )
                init_param(self.person_head.modules())

    def forward(
        self,
        x,
        grl_lambda=0.0,
        return_person=False,
        return_consistency=False,
    ):

        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)

        # input branches
        branch_features = [branch(x[i]) for i, branch in enumerate(self.input_branches)]
        if self.stream_fusion is None:
            x = torch.cat(branch_features, dim=1)
        else:
            x = self.stream_fusion(branch_features)
        if self.temporal_refine is not None:
            x = self.temporal_refine(x)

        # Optional paired-consistency features (disabled in GeoTri-GCN).
        # Pool time and joints before the shared stream; retain the person axis.
        fusion_feature = None
        if return_consistency:
            _, fusion_channel, fusion_time, fusion_joint = x.size()
            fusion_feature = x.view(
                N, M, fusion_channel, fusion_time, fusion_joint
            ).mean(dim=(3, 4)).permute(0, 2, 1)

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)

        # Optional paired-consistency logits bypass classifier dropout.
        # Action classification still uses the ordinary classifier above.
        consistency_out = None
        if return_consistency:
            pooled_feature = self.classifier.gap(feature)
            consistency_out = self.classifier.fc(pooled_feature).view(N, -1)

        if return_person:
            if self.person_head is None:
                raise RuntimeError('return_person=True requires person_adversarial=True')
            pooled_feature = feature.mean(dim=(2, 3, 4))
            person_out = self.person_head(pooled_feature, grl_lambda)
            if return_consistency:
                return (
                    out,
                    feature,
                    person_out,
                    fusion_feature,
                    consistency_out,
                )
            return out, feature, person_out

        if return_consistency:
            return out, feature, fusion_feature, consistency_out

        return out, feature


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
