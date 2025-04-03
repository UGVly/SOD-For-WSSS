import os
import torch
from datetime import datetime
import SemSegDataset

import logging
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from networks.models_config import parse_option
from networks.AnyNet import SOD_Net
from tqdm import tqdm
from tqdm.contrib import tenumerate
from loss import *
from utils import save_project,clip_gradient,linear_annealing
import pickle
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler
import sod_dataset
from torch.nn.functional import kl_div

args,config = parse_option()

# set the device for training
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('USE GPU:', args.gpu_id)


print(args)
print(config)

# # cam_list = list(set([cam[:11] for cam in os.listdir('/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/train/box')]))

# # print(len(cam_list))

# # cam_size = len(os.listdir('/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/train/cam'))
# # image_size = len(os.listdir('/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/train/image'))
# # box_size = len(os.listdir('/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/train/box'))
# # crop_size = len(os.listdir('/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/train/crop'))

# # print(cam_size,image_size,box_size,crop_size)

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

scaler = GradScaler()
model = SOD_Net(config)
model.cuda()

tri_loss = TripletLoss()


params = model.parameters()
optimizer = optim.Adam(params=params, lr = 1e-4, betas=[0.9,0.999], eps=1e-8)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 15, gamma= 0.5)


# log_path = args.log_path+ datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'AnyNet-'+args.backbone+'/'
# src_path = log_path+'src/'
# fig_path = log_path+'fig/'
# ckpt_path = log_path +'ckpt/'
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
#     os.makedirs(fig_path)
#     os.makedirs(ckpt_path)
#     os.makedirs(src_path)

#save src code and config
# save_project(src_path,['./','./networks'])

# with open(log_path+"args.json", mode="w") as f:
#     json.dump(args.__dict__, f, indent=4)

# config.dump(stream=open(log_path+"config.yaml", "w"))

train_loader = sod_dataset.get_weak_loader("/home/data4/ShiQiangShu/NAMSwinSegNet/dataset/", args.test_batch, config.DATA.IMG_SIZE,ds_type='train')


# total_step = len(train_loader)

# logging.basicConfig(filename=log_path+'log.txt', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
#                     level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# logging.info("Config:")
# logging.info(config)
# logging.info("Args:")
# logging.info(args)

step = 0

mae_list = []
miou_list = []


best_mae = 0
best_epoch = 0

class Feat_Similarity(nn.Module):
    def __init__(self):
        super(Feat_Similarity, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, input_feat, label):
        loss_sum = 0.0
        print(label.shape)
        for i in range(len(input_feat)):
            for j in range(i,len(input_feat)):
                loss_sum += kl_div(self.soft(input_feat[i]).log(), self.soft(input_feat[j])) if label[i] == label[j] \
                    else 1 - kl_div(self.soft(input_feat[i]).log(), self.soft(input_feat[j]))

        return  loss_sum

Feat_Sim = Feat_Similarity()

loss_fn = torch.nn.CrossEntropyLoss()

def func():
    pass

def train(train_loader, model, optimizer, epoch):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    #
    try:

        for i, (images, labels) in tenumerate(train_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda()
                gts = F.one_hot(labels,num_classes=20).cuda()
                s1,cls,f1 = model(images)
                
                
                loss = loss_fn(gts.float(),cls)+Feat_Sim(f1,labels)
        

            scaler.scale(loss).backward()  # 将张量乘以比例因子，反向传播
            clip_gradient(optimizer, args.clip) # 裁剪梯度
            scaler.step(optimizer)  # 将优化器的梯度张量除以比例因子。
            scaler.update()  # 更新比例因子

            step += 1
            epoch_step = epoch_step +1
            loss_all = loss_all + loss.item()
            # if i % 20 == 0 or i == total_step or i == 1:
            #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
            #           format(datetime.now(), epoch+1, args.max_epoch, i, total_step, loss.item()))
            #     logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
            #                  format(epoch+1, args.max_epoch, i, total_step, loss.item()))
                
        lr_scheduler.step()
        loss_all /= epoch_step
        #logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, LR: {}'.format(epoch+1, args.max_epoch, loss_all, optimizer.param_groups[0]['lr']))

        #if (epoch+1) % 10 == 0 or (epoch+1) == args.max_epoch:
        #    torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch+1))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        #torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# # val function
def val(val_loader, model, epoch):
    global best_mae, best_epoch
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        mae_loss = torch.nn.L1Loss()
        sum_IoU = 0.0
        sum_mae = 0.0
        for i, (images, gts) in tenumerate(val_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda()
                gts = gts.cuda()
                
                s1,s2,s3,s4 = model(images)
                
                s = s1.sigmoid()

            sum_mae += mae_loss(s,gts).item()*len(gts)
            sum_IoU += iou_loss(s,gts).item()

        
            mae_list.append(sum_mae/len(val_loader.dataset))
            miou_list.append(sum_IoU/len(val_loader.dataset))

        print("MIoU:",miou_list[-1])
        print("MAE:", mae_list[-1])

        if epoch == 0 or mae_list[-1] < best_mae:
            best_mae = mae_list[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), "./MultiLabelClassification/src/ckpt/Best_mae_test.pth")
            print('update best epoch to epoch {}'.format(epoch))
        
        print("Best MAE",best_mae)
        print("Best Epoch",best_epoch)


        
#         data = {
#         "MAE":mae_list,
#         "MIoU":miou_list,
#         }
#         plt.clf()
#         plt.plot(mae_list)
#         plt.savefig(fig_path+"mae.png")
#         plt.clf()
#         plt.plot(miou_list)
#         plt.savefig(fig_path+"miou.png")

#         pickle.dump(data,open(fig_path+'data.pkl','wb'))

def pretrain_clf(train_loader,model,optimizer,epoch):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    #
    try:
        for i, (images, gts) in tenumerate(train_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda()
                gts = gts.cuda()

                s1,cls,f1 = model(images)
                
                fa,fp,fn = func(f1)
                
                loss = loss_fn(gts,cls)+tri_loss(fa,fp,fn)
        
            scaler.scale(loss).backward()  # 将张量乘以比例因子，反向传播
            clip_gradient(optimizer, args.clip) # 裁剪梯度
            scaler.step(optimizer)  # 将优化器的梯度张量除以比例因子。
            scaler.update()  # 更新比例因子

            step += 1
            epoch_step = epoch_step +1
            loss_all = loss_all + loss.item()
                
        lr_scheduler.step()
        loss_all /= epoch_step
        #logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, LR: {}'.format(epoch+1, args.max_epoch, loss_all, optimizer.param_groups[0]['lr']))

        #if (epoch+1) % 10 == 0 or (epoch+1) == args.max_epoch:
        #    torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch+1))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        #torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

if __name__ == '__main__':



    for epoch in range(args.max_epoch):
        train(train_loader, model, optimizer, epoch)
        #val(val_loader, model, epoch)