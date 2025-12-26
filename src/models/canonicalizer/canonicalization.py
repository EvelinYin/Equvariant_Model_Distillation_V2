import torch.nn.functional as F
# import kornia as K
import torch
from .gcnn import LogitOverGroups
import matplotlib.pyplot as plt


class CanonicalizationNetwork(torch.nn.Module):
    def __init__(
        self,
        group,
        in_channels,
        out_channels,
        kernel_size,
        hidden_channel_list,
        dropout_p,
    ):
        super().__init__()
        self.group = group
        self.logit_over_groups = LogitOverGroups(
            group=group,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channel_list=hidden_channel_list,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        fibre_features = self.logit_over_groups(x).squeeze(1)
        canonicalized_images, indicator = self.get_canonicalized_images(x, fibre_features)
        canonicalized_images = canonicalized_images.cuda()
        return canonicalized_images, fibre_features, indicator

    def get_canonicalized_images(self, images, fibre_features):
        num_group_elements = fibre_features.shape[-1]
        # breakpoint()
        fibre_features_one_hot = F.one_hot(
            torch.argmax(fibre_features, dim=-1), num_group_elements
        ).float()

        fibre_features_soft = F.softmax(fibre_features, dim=-1)
        
        ref = self.group.get_canonicalization_ref(
            device=fibre_features_one_hot.device, dtype=fibre_features.dtype
        )


        indicator = torch.sum(
            (
                fibre_features_one_hot
                + fibre_features_soft
                - fibre_features_soft.detach()
            ) * ref,
            dim=-1,
        )

        canonicalized_images, indicator = self.group.get_canonicalized_images(images, indicator)


        return canonicalized_images, indicator