import os
import torch
from datetime import datetime
import SemSegDataset

import logging
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from networks.models_config import parse_option
from networks.AnyNet import AnyNet
from tqdm import tqdm
from loss import *
from utils import save_project,clip_gradient,linear_annealing
import pickle
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler

args,config = parse_option()

# set the device for training
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('USE GPU:', args.gpu_id)


print(args)
print(config)


scaler = GradScaler()
model = AnyNet(config)
model.cuda()

params = model.parameters()
optimizer = optim.Adam(params=params, lr = args.lr, betas=[0.9,0.999], eps=1e-8)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.decay_epoch*i for i in range(1,7)]+[120], gamma=args.gamma)

log_path = args.log_path+ datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'AnyNet-'+args.backbone+'/'
src_path = log_path+'src/'
fig_path = log_path+'fig/'
ckpt_path = log_path +'ckpt/'


# if not os.path.exists(log_path):
#     os.makedirs(log_path)
#     os.makedirs(fig_path)
#     os.makedirs(ckpt_path)
#     os.makedirs(src_path)

# #save src code and config
# save_project(src_path,['./','./networks'])

# with open(log_path+"args.json", mode="w") as f:
#     json.dump(args.__dict__, f, indent=4)

# config.dump(stream=open(log_path+"config.yaml", "w"))

train_loader = SemSegDataset.get_loader(args.train_root, args.train_batch, config.DATA.IMG_SIZE,False,args.texture,"FFTrans" in args.mfusion,"train")

#val_loader = SemSegDataset.get_loader(args.val_root, args.train_batch, config.DATA.IMG_SIZE,False,args.texture,"FFTrans" in args.mfusion,ds_type="val")

total_step = len(train_loader)

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



def train(train_loader, model, optimizer, epoch):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        anneal_reg = linear_annealing(0, 1, epoch, args.max_epoch)
        redu_reg = linear_annealing(1,0.5,epoch,args.max_epoch)
        for i, (images, gts,depths,texs,bounds) in enumerate(train_loader, start=1):
    
            with autocast():
                optimizer.zero_grad()
                images = images.cuda()
                depths = depths.cuda()
                gts = gts.cuda()
                texs = texs.cuda()
                bounds = bounds.cuda()
                s1, s2, s3, s4, edge1, edge2, latent_loss = model(images, depths)

                #loss5 = NAMLABSupervision(edge2,texs) if args.texture else 0.0

                loss7 = 0.1 * anneal_reg * latent_loss

                loss = loss7
                #loss = GTSupervision(s1,s2,s3,s4,gts,bounds,1,0) + \
                #       EdgeSupervision(edge1,bounds,1, args.edge_thickness if epoch < 10 else 0) +loss5 + loss7

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
        torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# val function
def val(val_loader, model, epoch):
    global best_mae, best_epoch

    model.eval()
    with torch.no_grad():
        mae_loss = torch.nn.L1Loss()
        sum_IoU = 0.0
        sum_mae = 0.0
        for _, (images, gts,depths) in enumerate(val_loader, start=1):
            
            gts = gts.cuda()
            images = images.cuda()
            depths = depths.cuda()

            res = model(images, depths)
            res = torch.sigmoid(res[0])

            sum_mae += mae_loss(res,gts).item()*len(gts)
            sum_IoU += IoU(res,gts).item()

        
        mae_list.append(sum_mae/len(val_loader.dataset))
        miou_list.append(sum_IoU/len(val_loader.dataset))

        print("MIoU:",miou_list[-1])
        print("MAE:", mae_list[-1])

        if epoch == 0 or mae_list[-1] < best_mae:
            best_mae = mae_list[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path + "Best_mae_test.pth")
            print('update best epoch to epoch {}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae_list[-1], best_epoch,best_mae ))
        print("Best MAE",best_mae)
        print("Best Epoch",best_epoch)
        
        data = {
        "MAE":mae_list,
        "MIoU":miou_list,
        }
        plt.clf()
        plt.plot(mae_list)
        plt.savefig(fig_path+"mae.png")
        plt.clf()
        plt.plot(miou_list)
        plt.savefig(fig_path+"miou.png")

        pickle.dump(data,open(fig_path+'data.pkl','wb'))


if __name__ == '__main__':

    for epoch in tqdm(range(args.max_epoch)):
        train(train_loader, model, optimizer, epoch)
        #val(val_loader, model, epoch)
    
