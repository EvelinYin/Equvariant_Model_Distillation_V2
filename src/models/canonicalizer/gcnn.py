import torch
from src.equ_lib.layers.lifting_convolution import LiftingConvolution
from src.equ_lib.layers.group_convolution import GroupConvolution
class LogitOverGroups(torch.nn.Module):

    def __init__(
        self,
        group,
        in_channels,
        out_channels,
        kernel_size,
        hidden_channel_list,
        dropout_p=0,
        padding=0,
    ):
        super().__init__()

        self.dropout_p = dropout_p

        self.lifting_conv = LiftingConvolution(
            in_channels=in_channels,
            out_channels=hidden_channel_list[0],
            kernel_size=kernel_size,
            bias=True,
            padding=padding,
            group=group
        )

        prev_channels = hidden_channel_list[0]

        hidden_channel_list = hidden_channel_list[1:] + [out_channels]

        self.gconvs = torch.nn.ModuleList()

        for i in range(len(hidden_channel_list)):
            self.gconvs.append(
                GroupConvolution(
                    in_channels=prev_channels,
                    out_channels=hidden_channel_list[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    group=group
                )
            )
            prev_channels = hidden_channel_list[i]

    def forward(self, x):
        x = self.lifting_conv(x)
        x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)

        for gconv in self.gconvs:
            x = gconv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
        # breakpoint()
        return torch.mean(x, dim=[-1, -2, -3])

        


