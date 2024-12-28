import torch
import torch.nn as nn

# input1 = torch.randn(1,512,12,8,12)
# input2 = torch.randn(1,256,24,16,24)
# input3 = torch.randn(1,128,48,32,48)
#
#
# avg_pool1 = nn.AvgPool3d(kernel_size=(1,8,1),stride=(1,8,1))
# avg_pool2 = nn.AvgPool3d(kernel_size=(1,16,1),stride=(1,16,1))
# avg_pool3 = nn.AvgPool3d(kernel_size=(1,32,1),stride=(1,32,1))
#
# output1=  avg_pool1(input1)
# output2=  avg_pool2(input2)
# output3=  avg_pool3(input3)
# print(11)

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

_dim_ = [128, 256, 512]


class FeatureTransformer(nn.Module):
    def __init__(self):
        super(FeatureTransformer, self).__init__()
        self.out_indices = [0, 2, 4, 6]
        self.out_indices = [0, 2, 4, 6]
        _dim_ = [128, 256, 512]
        upsample_strides = [1, 2, 1, 2, 1, 2, 1]
        in_channels = [_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64]
        out_channels = [256, _dim_[1], 128, _dim_[0], 64, 64, 32]

        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg = dict(type='deconv', bias=False)
        conv_cfg = dict(type='Conv2d', bias=False)

        self.deblocks2d = nn.ModuleList()
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            self.deblocks2d.append(deblock)

        self.avg_pool1 = nn.AvgPool3d(kernel_size=(1, 32, 1), stride=(1, 32, 1))
        self.avg_pool2 = nn.AvgPool3d(kernel_size=(1, 16, 1), stride=(1, 16, 1))
        self.avg_pool3 = nn.AvgPool3d(kernel_size=(1, 8, 1), stride=(1, 8, 1))
        # # 定义组归一化层，输入特征的通道数为512
        # self.group_norm = nn.GroupNorm(num_groups=32, num_channels=512,requires_grad=True)
        # # 定义二维卷积层，将通道数从512转换为256
        # self.conv2d = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, volume_embed_reshape):
        """
        :param x: 输入特征张量，形状为 [1, 512, 12, 1, 12]，
                  这里我们将其维度调整为 [1 * 12 * 12, 512] 以适配二维操作
        :return: 输出特征张量，形状为 [1, 256, 12, 1, 12]，
                  中间经过二维操作后再调整回原始维度形式
        """
        outputs = []
        volume_embed_reshape[0] = self.avg_pool1(volume_embed_reshape[0])
        batch_size, channels, dim1, dim2, dim3 = volume_embed_reshape[0].size()
        volume_embed_reshape[0] = volume_embed_reshape[0].view(batch_size, channels, dim1, dim3)

        volume_embed_reshape[1] = self.avg_pool2(volume_embed_reshape[1])
        batch_size, channels, dim1, dim2, dim3 = volume_embed_reshape[1].size()
        volume_embed_reshape[1] = volume_embed_reshape[1].view(batch_size, channels, dim1, dim3)

        volume_embed_reshape[2] = self.avg_pool3(volume_embed_reshape[2])
        batch_size, channels, dim1, dim2, dim3 = volume_embed_reshape[2].size()
        volume_embed_reshape[2] = volume_embed_reshape[2].view(batch_size, channels, dim1, dim3)
        result = volume_embed_reshape.pop()
        for i in range(len(self.deblocks2d)):
            result = self.deblocks2d[i](result)
            # print("resulut:")
            # print(result.shape)
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks2d) - 2:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp
        for i,feature in  enumerate(outputs):
            batch_size, channels, dim1,  dim3 = volume_embed_reshape[1].size()
            outputs[i] = feature.view(batch_size, channels, dim1, 1, dim3)
        # x = self.avg_pool1(x)
        # batch_size, channels, dim1, dim2, dim3 = x.size()
        # x = x.view(batch_size, channels, dim1, dim3)
        #
        # # 进行组归一化操作
        # x = self.group_norm(x)
        #
        # # 进行二维卷积操作
        # x = self.conv2d(x)

        return outputs


if __name__ == '__main__':
    out_indices = [0, 2, 4, 6]
    input1 = torch.randn(1, 128, 48, 32, 48)
    input2 = torch.randn(1, 256, 24, 16, 24)
    input3 = torch.randn(1, 512, 12, 8, 12)

    volume_embed_reshape = []

    volume_embed_reshape.append(input1)
    volume_embed_reshape.append(input2)
    volume_embed_reshape.append(input3)
    model = FeatureTransformer()
    output = model(volume_embed_reshape)
    print(111)
    # avg_pool1 = nn.AvgPool3d(kernel_size=(1,8,1),stride=(1,8,1))
    # avg_pool2 = nn.AvgPool3d(kernel_size=(1,16,1),stride=(1,16,1))
    # avg_pool3 = nn.AvgPool3d(kernel_size=(1,32,1),stride=(1,32,1))
    # input1 = avg_pool1(input1)
    # input1 = avg_pool1(input2)
    # input1 = avg_pool1(input3)

    #
    # result = volume_embed_reshape.pop()
    #
    # outputs = []
    # deblocks = YourModule()
    # for i in range(len(deblocks)):
    #     result = deblocks[i](result)
    #     # print("resulut:")
    #     # print(result.shape)
    #     if i in out_indices:
    #         outputs.append(result)
    #     elif i < len(deblocks) - 2:  # we do not add skip connection at level 0
    #         volume_embed_temp = volume_embed_reshape.pop()
    #         # print("xingzhuang")
    #         # print(volume_embed_temp.shape)
    #         # print("resultshape33")
    #         # print(result.shape)
    #         # print("vol")
    #         # print(volume_embed_temp.shape)
    #         result = result + volume_embed_temp
