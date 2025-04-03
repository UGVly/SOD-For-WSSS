import torch.nn as nn
from networks.BConv2d import *
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class UPFFT(nn.Module):
    def __init__(self,inchannel,outchannel,size):
        super().__init__()
       
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        # self.DWT =DTCWTForward(J=3, include_scale=True)
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(outchannel, outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel, outchannel)
        self.up = nn.Upsample(size*2)
        self.down = nn.Upsample(size)


    def forward(self, x, y):
        x = self.up(x)
        y = self.up(y)
        y = self.conv2(y)
        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)
        x_y = self.conv1(Xl) + self.conv1(Yl)

        x_m = self.IWT((x_y, Xh))
        y_m = self.IWT((x_y, Yh))

        out = self.conv3(x_m + y_m)
        
        return self.down(out)

class FTransFusion(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 act_layer=nn.GELU, drop=0.,enhance=False):
        super().__init__()
        self.out_features = in_features // 4
        hidden_features = hidden_features or in_features
        self.fuse_enhance = fuse_enhance(in_features) if enhance else nn.Identity()
        self.fc1 = nn.Conv2d(in_features*2, hidden_features*2, 1)
        self.act = act_layer()
        self.fc2 = TransBasicConv2d(hidden_features*2,self.out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, x):
        rgb,x = self.fuse_enhance((rgb,x))
        x = torch.cat((rgb,x),1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FFusion(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features,out_features, hidden_features=None, 
                 act_layer=nn.GELU, drop=0.,enhance = True,bn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fuse_enhance = fuse_enhance(in_features) if enhance else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features*2, 1)
        self.batchnorm = nn.BatchNorm2d(hidden_features) if bn else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features*2, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, x):
        rgb,x = self.fuse_enhance((rgb,x))
        x = torch.cat((rgb,x),1)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

    
class fuse_enhance(nn.Module):
    def __init__(self, infeature):
        super(fuse_enhance, self).__init__()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

    def forward(self,rd):
        r = rd[0]
        d = rd[1]
        assert r.shape == d.shape,"rgb and depth should have same size"

        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)

        r_out = r * r_ca
        d_out = d * d_ca
        return (r_out, d_out)

def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)

class LS_Fusion(nn.Module):
    def __init__(self,in_chans,out_chans):
        self.upsample_g = nn.Sequential(nn.Conv2d(in_chans, out_chans, 3, 1, 1, ), nn.BatchNorm2d(out_chans), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

    def forward(self,rgb,x):

        return self.upsample_g( rgb + x)


class CrossAttentionFusionPool(nn.Module):
    def __init__(self, channel, dilation, kernel=5):
        super(CrossAttentionFusionPool, self).__init__()
        self.spatial_att_1 = SpatialAttention()
        self.spatial_att_2 = SpatialAttention()
        self.channel_att_1 = ChannelAttention(channel=channel)
        self.channel_att_2 = ChannelAttention(channel=channel)
        self.pool_size = 2 * (kernel - 1) * dilation + 1
        self.pool1 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.pool2 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.gap = nn.AdaptiveMaxPool3d((2, 1, 1))
        # todo: condudct abalation analysis to find the optimal conv number
        self.d_conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation),
                      padding=(0, dilation, dilation),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation),
                      padding=(0, dilation, dilation),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=2)
        # self.rgb_refine = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     norm_layer(channel),
        #     nn.ReLU(inplace=True),
        # )
        # self.depth_refine = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     norm_layer(channel),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, rgb, depth):
        # cross attention
        rgb_1 = rgb * self.spatial_att_1(depth)
        depth_1 = depth * self.spatial_att_2(rgb)
        rgb_1 = rgb_1 * self.channel_att_1(rgb_1)
        depth_1 = depth_1 * self.channel_att_2(depth_1)

        rgb_2 = self.pool1(rgb_1)
        depth_2 = self.pool2(depth_1)

        rgb_2 = rgb_2.unsqueeze(2)
        depth_2 = depth_2.unsqueeze(2)
        f = torch.cat([rgb_2, depth_2], dim=2)
        f = self.d_conv1(self.d_conv2(f))
        f = self.gap(f)
        f = self.softmax(f)
        fused = f[:, :, 0, :, :] * rgb_1.squeeze(2) + f[:, :, 1, :, :] * depth_1.squeeze(2)
        #rgb_ret = rgb + self.rgb_refine(fused)
        #depth_ret = depth + self.depth_refine(fused)
        return fused#rgb_ret, depth_ret, fused

class HAIM(nn.Module):
    def __init__(self, in_channel):
        super(HAIM, self).__init__()
        self.relu = nn.ReLU(True)
        self.rgb_branch1 = BasicConv2d(in_channel, in_channel//4, 3, padding=1, dilation=1)
        self.rgb_branch2 = BasicConv2d(in_channel, in_channel//4, 3, padding=3, dilation=3)
        self.rgb_branch3 = BasicConv2d(in_channel, in_channel//4, 3, padding=5, dilation=5)
        self.rgb_branch4 = BasicConv2d(in_channel, in_channel//4, 3, padding=7, dilation=7)

        self.d_branch1 = BasicConv2d(in_channel, in_channel//4, 3, padding=1, dilation=1)
        self.d_branch2 = BasicConv2d(in_channel, in_channel//4, 3, padding=3, dilation=3)
        self.d_branch3 = BasicConv2d(in_channel, in_channel//4, 3, padding=5, dilation=5)
        self.d_branch4 = BasicConv2d(in_channel, in_channel//4, 3, padding=7, dilation=7)

        self.rgb_branch1_sa = SpatialAttention()
        self.rgb_branch2_sa = SpatialAttention()
        self.rgb_branch3_sa = SpatialAttention()
        self.rgb_branch4_sa = SpatialAttention()

        self.rgb_branch1_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch2_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch3_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch4_ca = ChannelAttention(in_channel // 4)

        self.r_branch1_sa = SpatialAttention()
        self.r_branch2_sa = SpatialAttention()
        self.r_branch3_sa = SpatialAttention()
        self.r_branch4_sa = SpatialAttention()

        self.r_branch1_ca = ChannelAttention(in_channel // 4)
        self.r_branch2_ca = ChannelAttention(in_channel // 4)
        self.r_branch3_ca = ChannelAttention(in_channel // 4)
        self.r_branch4_ca = ChannelAttention(in_channel // 4)

        self.ca = ChannelAttention(in_channel)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_rgb, x_d):
        x1_rgb = self.rgb_branch1(x_rgb)
        x2_rgb = self.rgb_branch2(x_rgb)
        x3_rgb = self.rgb_branch3(x_rgb)
        x4_rgb = self.rgb_branch4(x_rgb)

        x1_d = self.d_branch1(x_d)
        x2_d = self.d_branch2(x_d)
        x3_d = self.d_branch3(x_d)
        x4_d = self.d_branch4(x_d)

        x1_rgb_ca = x1_rgb.mul(self.rgb_branch1_ca(x1_rgb))
        x1_d_sa = x1_d.mul(self.rgb_branch1_sa(x1_rgb_ca))
        x1_d = x1_d + x1_d_sa
        x1_d_ca = x1_d.mul(self.r_branch1_ca(x1_d))
        x1_rgb_sa = x1_rgb.mul(self.r_branch1_sa(x1_d_ca))
        x2_rgb = x2_rgb + x1_rgb_sa

        x2_rgb_ca = x2_rgb.mul(self.rgb_branch2_ca(x2_rgb))
        x2_d_sa = x2_d.mul(self.rgb_branch2_sa(x2_rgb_ca))
        x2_d = x2_d + x2_d_sa
        x2_d_ca = x2_d.mul(self.r_branch2_ca(x2_d))
        x2_rgb_sa = x2_rgb.mul(self.r_branch2_sa(x2_d_ca))
        x3_rgb = x3_rgb + x2_rgb_sa

        x3_rgb_ca = x3_rgb.mul(self.rgb_branch3_ca(x3_rgb))
        x3_d_sa = x3_d.mul(self.rgb_branch3_sa(x3_rgb_ca))
        x3_d = x3_d + x3_d_sa
        x3_d_ca = x3_d.mul(self.r_branch3_ca(x3_d))
        x3_rgb_sa = x3_rgb.mul(self.r_branch3_sa(x3_d_ca))
        x4_rgb = x4_rgb + x3_rgb_sa

        x4_rgb_ca = x4_rgb.mul(self.rgb_branch4_ca(x4_rgb))
        x4_d_sa = x4_d.mul(self.rgb_branch4_sa(x4_rgb_ca))
        x4_d = x4_d + x4_d_sa
        x4_d_ca = x4_d.mul(self.r_branch4_ca(x4_d))
        x4_rgb_sa = x4_rgb.mul(self.r_branch4_sa(x4_d_ca))


        y = torch.cat((x1_rgb_sa, x2_rgb_sa, x3_rgb_sa, x4_rgb_sa), 1)
        y_ca = y.mul(self.ca(y))

        z = y_ca + x_rgb
        # then try z = y_ca + x_rgb + x_d, choose the better performance

        return z

#-----------

class EncDecFusing(nn.Module):
    def __init__(self, in_channels):
        super(EncDecFusing, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fusing_layer = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, enc_fea, dec_fea):
        enc_fea = self.enc_fea_proc(enc_fea)

        if dec_fea.size(2) != enc_fea.size(2):
            dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear',
                                 align_corners=True)

        enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.fusing_layer(enc_fea)
        return output
    

#-------

class PATM(nn.Module):

    def __init__(self,in_chan,out_chan):
        super(PATM, self).__init__()
        self.conv = conv3x3_bn_relu(in_chan, out_chan)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 4),
            conv3x3(32, 1)
        )
        self.relu = nn.ReLU(True)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.conv64_1 = conv3x3(64, 1)

    def forward(self,fuse_fea,edge_fea):
        end_sal = self.conv(fuse_fea)
        
        out = self.relu(torch.cat((end_sal, edge_fea), dim=1))
        out = self.up4(out)
        return self.conv64_1(out)
    
def build_modilty_fusion(config,embed_dims):
    fusion_type = config.MODEL.MFUSION
    
    if fusion_type == "FF-base":
        ff1 = FFusion(embed_dims[0],embed_dims[0],enhance=False)
        ff2 = FFusion(embed_dims[1],embed_dims[1],enhance=False)
        ff3 = FFusion(embed_dims[2],embed_dims[2],enhance=False)
        ff4 = FFusion(embed_dims[3],embed_dims[3],enhance=False)
        return embed_dims,ff1,ff2,ff3,ff4
    elif fusion_type == "FF-enhance":
        ff1 = FFusion(embed_dims[0],embed_dims[0])
        ff2 = FFusion(embed_dims[1],embed_dims[1])
        ff3 = FFusion(embed_dims[2],embed_dims[2])
        ff4 = FFusion(embed_dims[3],embed_dims[3])
        return embed_dims,ff1,ff2,ff3,ff4        
    elif fusion_type == "FFTrans-base":
        ff1 = FTransFusion(embed_dims[0],enhance=False)
        ff2 = FTransFusion(embed_dims[1],enhance=False)
        ff3 = FTransFusion(embed_dims[2],enhance=False)
        ff4 = FTransFusion(embed_dims[3],enhance=False)
        return [dim//4 for dim in embed_dims],ff1,ff2,ff3,ff4
    elif fusion_type == "FFTrans-enhance":
        ff1 = FTransFusion(embed_dims[0])
        ff2 = FTransFusion(embed_dims[1])
        ff3 = FTransFusion(embed_dims[2])
        ff4 = FTransFusion(embed_dims[3])
        return [dim//4 for dim in embed_dims],ff1,ff2,ff3,ff4
    elif fusion_type == "UPFTT":
        ff1 = UPFFT(embed_dims[0],embed_dims[0])
        ff2 = UPFFT(embed_dims[1],embed_dims[1])
        ff3 = UPFFT(embed_dims[2],embed_dims[2])
        ff4 = UPFFT(embed_dims[3],embed_dims[3])
        return [dim//2 for dim in embed_dims],ff1,ff2,ff3,ff4
    elif fusion_type == "CAttenFP":
        ff1 = CrossAttentionFusionPool(embed_dims[0], 2, 5)
        ff2 = CrossAttentionFusionPool(embed_dims[1], 2, 5)
        ff3 = CrossAttentionFusionPool(embed_dims[2], 2, 5)
        ff4 = CrossAttentionFusionPool(embed_dims[3], 2, 5)
        return embed_dims,ff1,ff2,ff3,ff4
    elif fusion_type == "HAIM":
        ff1 = HAIM(embed_dims[0])
        ff2 = HAIM(embed_dims[1])
        ff3 = HAIM(embed_dims[2])
        ff4 = HAIM(embed_dims[3])
        return embed_dims,ff1,ff2,ff3,ff4
    else:
        raise NotImplementedError("no such modality fusion module")