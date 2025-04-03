import os
from PIL import Image,ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import torch.nn.functional as F
import pickle
import torch

# several data augumentation strategies
def random_flip(*images):
    if random.randint(0, 1):
        images = (image.transpose(Image.FLIP_LEFT_RIGHT) for image in images)

    if random.randint(0,1):
        images = (image.transpose(Image.FLIP_TOP_BOTTOM) for image in images)
    return images


def random_crop(*images):
    image = images[0]
    image_width,image_height = image.size[0], image.size[1]
    border_width,border_height = image_width*0.1,image_height*0.1
    crop_win_width = np.random.randint(image_width - border_width, image_width)
    crop_win_height = np.random.randint(image_height - border_height, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    images = (image.crop(random_region) for image in images)
    return images

def random_rotation(*images):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        images = (image.rotate(random_angle, Image.BICUBIC) for image in images)
        
    return images

def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def image_suffix(f):
    return f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')

class SalObjTrainDataset(data.Dataset):
    def __init__(self, dataset_root, texture_type=None, trainsize=224):

        image_root = dataset_root + '/imgs/'
        box_root = dataset_root + '/box/'
        if texture_type:
            texture_root = dataset_root + texture_type
        else:
            texture_root = dataset_root + '/bound/'
        gt_root = dataset_root + '/gt/'

    
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.texs = sorted([texture_root+f for f in os.listdir(texture_root) if image_suffix(f)])
        
    

        assert len(self.images) == len(self.texs) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} train data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.logistic_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        texture = self.binary_loader(self.texs[index])

        image, gt, texture = random_flip(image, gt, texture )
        image, gt, texture = random_crop(image, gt, texture)
        image, gt, texture = random_rotation(image, gt, texture)
        image = color_enhance(image)
        
        
        image = self.rgb_transform(image)
        gt = self.binary_transform(gt)
        texture = self.binary_transform(texture)
        
        texture = F.avg_pool2d(F.max_pool2d(texture,kernel_size=3,stride=1,padding=1),kernel_size=3,stride=1,padding=1)
        return image, gt, texture


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')


    def __len__(self):
        return self.size

class SalObjValDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize=224):

        image_root = dataset_root + '/imgs/'
        gt_root = dataset_root + '/gt/'
        
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])

        
        assert len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} val data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()]) 


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        
        gt = self.binary_loader(self.gts[index])

        image = self.rgb_transform(image)
        gt = self.binary_transform(gt)
        
        return image, gt


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size


def fname2shape(fname):
    s = fname.split('_')
    return (int(s[6])-int(s[4]),int(s[5])-int(s[3]))

def get_sod_loader(dataset_root, batchsize, trainsize, dist = False, texture_type = None, ds_type='train'):
    if ds_type == 'train':   
        train_dataset = data.ConcatDataset([SalObjTrainDataset(os.path.join(dataset_root,dataset_name),texture_type,trainsize) \
                                            for dataset_name in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      shuffle=True,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        
        return data_loader
    
    elif ds_type == 'val':

        dataset = data.ConcatDataset([SalObjValDataset(os.path.join(dataset_root,dataset_name),trainsize) \
                                            for dataset_name in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      num_workers=4)
        
        return data_loader
    else:
        raise NotImplementedError("no such dataset")



    
if __name__ == "__main__":
    train_loader = get_sod_loader('./dataset/SOD/train',8,224,texture_type='/_namlab30/',ds_type='train')

    for i, (images, gts,texs) in enumerate(train_loader, start=1):
       print(images.shape,gts.shape,texs.shape)
    