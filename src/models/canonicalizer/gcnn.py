import torch
from src.equ_lib.layers.lifting_convolution import LiftingConvolution
from src.equ_lib.layers.group_convolution import GroupConvolution
class LogitOverGroups(torch.nn.Module):

    def __init__(
        self,
        use_equ_layers,
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
        self.use_equ_layers = use_equ_layers

        if use_equ_layers:
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
            
        else:
            prev_channels = in_channels
            hidden_channel_list = hidden_channel_list + [out_channels]


        self.gconvs = torch.nn.ModuleList()

        for i in range(len(hidden_channel_list)):
            if use_equ_layers:
                self.gconvs.append(
                    GroupConvolution(
                        in_channels=prev_channels,
                        out_channels=hidden_channel_list[i],
                        kernel_size=kernel_size,
                        padding=padding,
                        group=group
                    )
                )
            else:
                self.gconvs.append(
                    torch.nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=hidden_channel_list[i],
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
            prev_channels = hidden_channel_list[i]

    def forward(self, x):
        

        if self.use_equ_layers:
            norm_over_dim = -4
            x = self.lifting_conv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
        else:
            norm_over_dim = -3  

        for gconv in self.gconvs:
            x = gconv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[norm_over_dim:])
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)

        if self.use_equ_layers:
            return torch.mean(x, dim=[-1, -2, -3])
        else:
            return torch.mean(x, dim=[-2, -1])


        


