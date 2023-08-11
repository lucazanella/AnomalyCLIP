import torch
import torch.nn as nn
import torch.nn.init as torch_init

torch.set_default_tensor_type("torch.FloatTensor")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class Aggregate(nn.Module):
    def __init__(
        self,
        len_feature: int = 512,
        reduction: int = 4,
    ):
        super().__init__()

        dim_inner = len_feature // reduction

        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=dim_inner,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.ReLU(),
            bn(dim_inner)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=dim_inner,
                kernel_size=3,
                stride=1,
                dilation=2,
                padding=2,
            ),
            nn.ReLU(),
            bn(dim_inner)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=dim_inner,
                kernel_size=3,
                stride=1,
                dilation=4,
                padding=4,
            ),
            nn.ReLU(),
            bn(dim_inner)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=dim_inner,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=len_feature,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(dim_inner, sub_sample=False, bn_layer=True)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)  # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)

        return out
