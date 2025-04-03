import sys
sys.path.append('/home/data1/ShiqiangShu/NAMSwinNet/')
from networks.PWNet import *
from networks.SwinNets import build_backbone
from networks.models_config import parse_option
import numpy as np
from networks.MFusionToolBox import *
from networks.EdgeAwareToolBox import *
from collections import OrderedDict

"""
TORCH_DISTRIBUTED_DEBUG
"""

def get_parameter_num(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return ('Trainable Parameters: %.3fM' % parameters)



class NAMPlusNetB(nn.Module):
    def __init__(self,config):
        super(NAMPlusNetB, self).__init__()

        self.encoderR, embed_dims = build_backbone(config)#swin_like_model

        self.encoderD, embed_dims = build_backbone(config)

        input_size = config.DATA.IMG_SIZE
        
        # Lateral layers
        self.lateral_conv0 = BasicConv2d(embed_dims[0], embed_dims[0], 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(embed_dims[2], embed_dims[1], 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)


        fused_dims,self.FFT1, self.FFT2, self.FFT3, self.FFT4 = build_modilty_fusion(config,embed_dims)

        self.conv_emb3_0 = BasicConv2d(fused_dims[3], fused_dims[0])
        self.conv_emb2_0 = BasicConv2d(fused_dims[2], fused_dims[0])
        self.conv_emb1_0 = BasicConv2d(fused_dims[1], fused_dims[0])

        self.S4 = nn.Conv2d(fused_dims[3], 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(fused_dims[2], 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(fused_dims[1], 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(fused_dims[0], 1, 3, stride=1, padding=1)
        
        
        self.up1 = nn.Upsample(input_size)
        self.up2 = nn.Upsample(input_size)
        self.up3 = nn.Upsample(input_size)
        self.up_loss = nn.Upsample(input_size//4)
        
        # Mutual_info_reg1
        self.mi_level1 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        self.mi_level2 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        self.mi_level3 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        self.mi_level4 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)

        
        #self.sod_edge_aware = Edge_Aware(embed_dims,input_size)
        self.sod_edge_aware = Edge_Module(embed_dims)
        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        
        #self.rgb_edge_aware = Edge_Aware(embed_dims,input_size)
        self.rgb_edge_aware = Edge_Module(embed_dims)
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

    def forward(self, x_rgb,x_thermal):
        x0,x1,x2,x3 = self.encoderR(x_rgb)        
        y0,y1,y2,y3 = self.encoderD(x_thermal)

        """
        [16, 128, 56, 56]
        [16, 256, 28, 28]
        [16, 512, 14, 14]
        [16, 1024, 7, 7 ]
        """

        x2_ACCoM = self.FFT1(x0, y0)
        x3_ACCoM = self.FFT2(x1, y1)
        x4_ACCoM = self.FFT3(x2, y2)
        x5_ACCoM = self.FFT4(x3, y3)
        
        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        #edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        edge_sod = self.sod_edge_aware(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

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

        s1 = self.up1(self.S1(mer_cros1))
        s2 = self.up2(self.S2(mer_cros2))
        s3 = self.up3(self.S3(mer_cros3))
        s4 = self.up3(self.S4(mer_cros4))


        x_loss0 = x0
        y_loss0 = y0
        x_loss1 = self.up_loss(self.conv_emb1_0(x1))
        y_loss1 = self.up_loss(self.conv_emb1_0(y1))
        x_loss2 = self.up_loss(self.conv_emb2_0(x2))
        y_loss2 = self.up_loss(self.conv_emb2_0(y2))
        x_loss3 = self.up_loss(self.conv_emb3_0(x3))
        y_loss3 = self.up_loss(self.conv_emb3_0(y3))

        lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1,s2,s3,s4,torch.sigmoid(edge_sod),torch.sigmoid(edge_rgb),lat_loss

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))
            
    def load_from_NetA(self,ckptA):
        encoderD_dict = OrderedDict()

        for k,v in ckptA.items():
            if k.startswith("encoderR"):
                encoderD_dict[k.replace("encoderR","encoderD")] = v
                encoderD_dict[k] = v
            elif k in self.state_dict().keys():
                encoderD_dict[k] = v
        print("convert yo net B")
        self.load_state_dict(encoderD_dict,strict=True)
        """
        self.lateral_conv0 = BasicConv2d(embed_dims[0], embed_dims[0], 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(embed_dims[2], embed_dims[1], 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)


        fused_dims,self.FFT1, self.FFT2, self.FFT3, self.FFT4 = build_modilty_fusion(config,embed_dims)

        #self.conv_emb3_0 = BasicConv2d(fused_dims[3], fused_dims[0])
        #self.conv_emb2_0 = BasicConv2d(fused_dims[2], fused_dims[0])
        #self.conv_emb1_0 = BasicConv2d(fused_dims[1], fused_dims[0])

        self.S4 = nn.Conv2d(fused_dims[3], 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(fused_dims[2], 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(fused_dims[1], 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(fused_dims[0], 1, 3, stride=1, padding=1)
        
        
        self.up1 = nn.Upsample(input_size)
        self.up2 = nn.Upsample(input_size)
        self.up3 = nn.Upsample(input_size)
        #self.up_loss = nn.Upsample(input_size//4)
        
        # Mutual_info_reg1
        #self.mi_level1 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        #self.mi_level2 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        #self.mi_level3 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)
        #self.mi_level4 = Mutual_info_reg(embed_dims[0], embed_dims[0], 6)

        
        self.sod_edge_aware = Edge_Aware(embed_dims,input_size)
        #self.sod_edge_aware = Edge_Module(embed_dims)

        
        self.rgb_edge_aware = Edge_Aware(embed_dims,input_size)
        #self.rgb_edge_aware = Edge_Module(embed_dims)
        #self.depth_edge_aware = Edge_Aware(embed_dims,input_size)




        # different scale feature fusion
        self.PATM4 = PATM_BAB(embed_dims[3], embed_dims[3], embed_dims[3], 3, 2)#TODO reduce params
        self.PATM3 = PATM_BAB(embed_dims[2]+embed_dims[3], embed_dims[3], embed_dims[2], 3, 2)
        self.PATM2 = PATM_BAB(embed_dims[1]+embed_dims[2], embed_dims[2], embed_dims[1], 5, 3)
        self.PATM1 = PATM_BAB(
        """

            


if __name__ == "__main__":
    
    args,config = parse_option()

    model = NAMPlusNetB(config)

    ckptA = torch.load('/home/data1/ShiqiangShu/NAMSwinNet/log/2024-01-23-22:41:46AnyNet-swin-base/ckpt/Best_mae_test.pth')

    model.load_from_NetA(ckptA)
    

    rgb = torch.randn(4,3,224,224)
    x = torch.randn(4,3,224,224)

    pred = model(rgb,x)
    print(pred[0].shape)


