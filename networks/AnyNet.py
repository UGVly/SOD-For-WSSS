import sys
sys.path.append('../')
sys.path.append('./')


from networks.PWNet import *
from networks.SwinNets import build_backbone
from networks.models_config import parse_option
import numpy as np
from networks.MFusionToolBox import *
from networks.EdgeAwareToolBox import *
from einops import rearrange

"""
TORCH_DISTRIBUTED_DEBUG
"""

def get_parameter_num(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return ('Trainable Parameters: %.3fM' % parameters)


class Interpolate(nn.Module):
    def __init__(self, size, mode = 'nearest'):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def forward(self, x):
        x = self.interpolate(x, size=self.size, mode=self.mode)
        return x


class AnyNet(nn.Module):
    def __init__(self,config):
        super(AnyNet, self).__init__()

        self.encoderR, embed_dims = build_backbone(config)#swin_like_model
        
        input_size = config.DATA.IMG_SIZE

        fused_dims = embed_dims
        
        self.FFT1 = FFusion(embed_dims[0]+16,embed_dims[0],enhance=False)
        self.FFT2 = FFusion(embed_dims[1]+64,embed_dims[1],enhance=False)
        self.FFT3 = FFusion(embed_dims[2]+256,embed_dims[2],enhance=False)
        self.FFT4 = FFusion(embed_dims[3]+1024,embed_dims[3],enhance=False)


        #self.conv_emb3_0 = BasicConv2d(embed_dims[3], embed_dims[0])
        #self.conv_emb2_0 = BasicConv2d(embed_dims[2], embed_dims[0])
        #self.conv_emb1_0 = BasicConv2d(embed_dims[1], fused_dims[0])

        self.S4 = nn.ConvTranspose2d(fused_dims[3], 1, 2, stride=2)
        self.S3 = nn.ConvTranspose2d(fused_dims[2], 1, 2, stride=2)
        self.S2 = nn.ConvTranspose2d(fused_dims[1], 1, 2, stride=2)
        self.S1 = nn.ConvTranspose2d(fused_dims[0], 1, 2, stride=2)
        

        #self.up_loss = Interpolate(size=input_size//4)
        #self.sod_edge_aware = Edge_Aware(fused_dims,input_size*2)
        #self.sod_edge_aware = Edge_Module(fused_dims)
        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        
        self.rgb_edge_aware = Edge_Aware(embed_dims,input_size*2)
        #self.rgb_edge_aware = Edge_Module(embed_dims)
        #self.depth_edge_aware = Edge_Aware(embed_dims,input_size)


        """
        (1024, 1024, 1024, 3, 2)
        (1536, 1024, 512, 3, 2)
        (768, 512, 256, 5, 3)
        (384, 256, 128, 5, 3)
        """

        # different scale feature fusion
        self.PATM4 = PATM_BAB(fused_dims[3], fused_dims[3], fused_dims[3], 3, 2)#TODO reduce params
        self.PATM3 = PATM_BAB(fused_dims[2]+fused_dims[3], fused_dims[3], fused_dims[2], 3, 2)
        self.PATM2 = PATM_BAB(fused_dims[1]+fused_dims[2], fused_dims[2], fused_dims[1], 5, 3)
        self.PATM1 = PATM_BAB(fused_dims[0]+fused_dims[1], fused_dims[1], fused_dims[0], 5, 3)
        
                # Mutual_info_reg1
        #self.mi_level1 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level2 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level3 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level4 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)

    def forward(self, rgb, x):
        x0,x1,x2,x3 = self.encoderR(rgb)        
        
        B,C,H,W = x.shape

        y0 = rearrange(x, 'b c (h0 p1) (w0 p2) -> b (c p1 p2) h0 w0',p1=4,p2=4,h0=H//4,w0=W//4)
        y1 = rearrange(x, 'b c (h1 p1) (w1 p2) -> b (c p1 p2) h1 w1',p1=8,p2=8,h1=H//8,w1=W//8)
        y2 = rearrange(x, 'b c (h2 p1) (w2 p2) -> b (c p1 p2) h2 w2',p1=16,p2=16,h2=H//16,w2=W//16)
        y3 = rearrange(x, 'b c (h3 p1) (w3 p2) -> b (c p1 p2) h3 w3',p1=32,p2=32,h3=H//32,w3=W//32)
        #y0,y1,y2,y3 = self.encoderR(self.depth_scale(x))
        """
        [16, 128, 56, 56]
        [16, 256, 28, 28]
        [16, 512, 14, 14]
        [16, 1024, 7, 7 ]

        [16, 16, 56, 56]
        [16, 64, 28, 28]
        [16, 256, 14, 14]
        [16, 1024, 7, 7 ]        
        """

        x2_ACCoM = self.FFT1(x0, y0)
        x3_ACCoM = self.FFT2(x1, y1)
        x4_ACCoM = self.FFT3(x2, y2)
        x5_ACCoM = self.FFT4(x3, y3)
        
        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        #edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        #edge_sod = self.sod_edge_aware(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)
    
        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)

        """
        torch.Size([16, 128, 112, 112]) torch.Size([16, 1024, 7, 7])
        torch.Size([16, 256, 56, 56]) torch.Size([16, 1536, 14, 14])
        torch.Size([16, 512, 28, 28]) torch.Size([16, 768, 28, 28])
        torch.Size([16, 1024, 14, 14]) torch.Size([16, 384, 56, 56])
        """

        s1 = self.S1(mer_cros1)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)



        # x_loss0 = x0
        # y_loss0 = y0
        # x_loss1 = self.up_loss(self.conv_emb1_0(x1))
        # y_loss1 = self.up_loss(self.conv_emb1_0(y1))
        # x_loss2 = self.up_loss(self.conv_emb2_0(x2))
        # y_loss2 = self.up_loss(self.conv_emb2_0(y2))
        # x_loss3 = self.up_loss(self.conv_emb3_0(x3))
        # y_loss3 = self.up_loss(self.conv_emb3_0(y3))

        # lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        # lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        # lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        # lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        #lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1,s2,s3,s4,edge_rgb#,lat_loss

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))
            

class SOD_A_Net(nn.Module):
    def __init__(self,config):
        super(SOD_A_Net, self).__init__()

        self.encoderR, embed_dims = build_backbone(config)#swin_like_model
        
        input_size = config.DATA.IMG_SIZE

        print(input_size)

        self.S4 = nn.ConvTranspose2d(embed_dims[3], 1, 2, stride=2)
        self.S3 = nn.ConvTranspose2d(embed_dims[2], 1, 2, stride=2)
        self.S2 = nn.ConvTranspose2d(embed_dims[1], 1, 2, stride=2)
        self.S1 = nn.ConvTranspose2d(embed_dims[0], 1, 2, stride=2)
        

        self.up_loss = Interpolate(size=input_size)
        #self.sod_edge_aware = Edge_Aware(fused_dims,input_size*2)
        #self.sod_edge_aware = Edge_Module(fused_dims)
        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        
        self.rgb_edge_aware = Edge_Aware(embed_dims,input_size)
        #self.rgb_edge_aware = Edge_Module(embed_dims)
        #self.depth_edge_aware = Edge_Aware(embed_dims,input_size)


        """
        (1024, 1024, 1024, 3, 2)
        (1536, 1024, 512, 3, 2)
        (768, 512, 256, 5, 3)
        (384, 256, 128, 5, 3)
        """

        # different scale feature fusion
        self.PATM4 = PATM_BAB(embed_dims[3], embed_dims[3], embed_dims[3], 3, 2)#TODO reduce params
        self.PATM3 = PATM_BAB(embed_dims[2]+embed_dims[3], embed_dims[3], embed_dims[2], 3, 2)
        self.PATM2 = PATM_BAB(embed_dims[1]+embed_dims[2], embed_dims[2], embed_dims[1], 5, 3)
        self.PATM1 = PATM_BAB(embed_dims[0]+embed_dims[1], embed_dims[1], embed_dims[0], 5, 3)
        
                # Mutual_info_reg1
        #self.mi_level1 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level2 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level3 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)
        #self.mi_level4 = Mutual_info_reg(fused_dims[0], fused_dims[0], 6)

    def forward(self, rgb):
        x0,x1,x2,x3 = self.encoderR(rgb)        
        
        #y0,y1,y2,y3 = self.encoderR(self.depth_scale(x))
        """
        [16, 128, 56, 56]
        [16, 256, 28, 28]
        [16, 512, 14, 14]
        [16, 1024, 7, 7 ]

        [16, 16, 56, 56]
        [16, 64, 28, 28]
        [16, 256, 14, 14]
        [16, 1024, 7, 7 ]        
        """
        
        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        #edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        #edge_sod = self.sod_edge_aware(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x3)
    
        m4 = torch.cat((mer_cros4,x2),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x1), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x0), dim=1)
        mer_cros1 = self.PATM1(m2)

        """
        torch.Size([16, 128, 112, 112]) torch.Size([16, 1024, 7, 7])
        torch.Size([16, 256, 56, 56]) torch.Size([16, 1536, 14, 14])
        torch.Size([16, 512, 28, 28]) torch.Size([16, 768, 28, 28])
        torch.Size([16, 1024, 14, 14]) torch.Size([16, 384, 56, 56])
        """

        s1 = self.up_loss(self.S1(mer_cros1))
        s2 = self.up_loss(self.S2(mer_cros2))
        s3 = self.up_loss(self.S3(mer_cros3))
        s4 = self.up_loss(self.S4(mer_cros4))



        # x_loss0 = x0
        # y_loss0 = y0
        # x_loss1 = self.up_loss(self.conv_emb1_0(x1))
        # y_loss1 = self.up_loss(self.conv_emb1_0(y1))
        # x_loss2 = self.up_loss(self.conv_emb2_0(x2))
        # y_loss2 = self.up_loss(self.conv_emb2_0(y2))
        # x_loss3 = self.up_loss(self.conv_emb3_0(x3))
        # y_loss3 = self.up_loss(self.conv_emb3_0(y3))

        # lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        # lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        # lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        # lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        #lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1,s2,s3,s4,edge_rgb#,lat_loss

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))


if __name__ == "__main__":
    
    args,config = parse_option()

    model = SOD_A_Net(config)

    rgb = torch.randn(4,3,224,224)
    
    pred = model(rgb)

    print(pred[0].shape,pred[1].shape,pred[2].shape,pred[3].shape,pred[4].shape)


