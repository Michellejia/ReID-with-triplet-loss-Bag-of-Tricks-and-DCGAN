#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools

from backbone import EmbedNetwork, ClassBlock
from loss import TripletLoss, CenterLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger
from torch.nn import init
from torchvision import datasets, models, transforms
from random_erasing import RandomErasing


transform_train_list = [
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.Resize((384,192), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0])
    ]
transform_val_list = [
    transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

#将多个transform组合起来使用
data_transforms = {
    'train': transforms.Compose( transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


train_all = '_all'

data_dir = "E://graduation thesis//Spatial-Temporal-Re-identification-master//Spatial-Temporal-Re-identification-master//dataset//market_rename"

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def train():
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    net = EmbedNetwork().cuda()
    net = nn.DataParallel(net)
    print(net)
    triplet_loss = TripletLoss(margin = 0.3).cuda() # no margin means soft-margin
    BNNeck = ClassBlock(2048, 1501).cuda()
    BNNeck = nn.DataParallel(BNNeck)

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(net.parameters(), lr = 3e-4, wd = 0, t0 = 15000, t1 = 25000)

    ## dataloader
    selector = BatchHardTripletSelector()
    ds = Market1501('E://graduation thesis//triplet-reid-pytorch-master//triplet-reid-pytorch-master//datasets//Market-1501-v15.09.15//Market-1501-v15.09.15//bounding_box_train', is_train = True)
    sampler = BatchSampler(ds, 9, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 0)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    loss1_avg = []
    loss2_avg = []
    loss3_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)

        #criterion = nn.CrossEntropyLoss().cuda()
        criterion = CrossEntropyLabelSmooth(num_classes=1501)
        center_criterion = CenterLoss(num_classes=1051, feat_dim=2048, use_gpu=True)

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()

        # for name in net.state_dict():
        #     print("net parameters:", name)

        optim.zero_grad()

        embds = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)
        
        BNNeck.train()
        # for name in BNNeck.state_dict():
        #     print("BNNeck parameters:", name)

        #print(BNNeck)

        classifier = []
        classifier += [nn.Linear(2048, 1501)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        classifier = classifier.cuda()

        x = torch.squeeze(embds)
        BNNeck1 = BNNeck(x)
        #classifier = torch.autograd.Variable(classifier)
        classifier = classifier(BNNeck1)

        loss1 = triplet_loss(anchor, positives, negatives)
        loss2 = criterion(classifier, lbs)
        loss3 = center_criterion(embds, lbs)
        loss = loss1 + loss2 + 0.0005*loss3
        
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        loss1_avg.append(loss1.detach().cpu().numpy())
        loss2_avg.append(loss2.detach().cpu().numpy())
        loss3_avg.append(loss3.detach().cpu().numpy())
        #loss1_avg.append(loss1.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss1_avg = sum(loss1_avg) / len(loss1_avg)
            loss2_avg = sum(loss2_avg) / len(loss2_avg)
            loss3_avg = sum(loss3_avg) / len(loss3_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss1: {:4f}, loss2: {:4f}, loss3:{:4f}, loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss1_avg, loss2_avg, loss3_avg, loss_avg, optim.lr, time_interval))
            loss_avg = []
            loss1_avg = []
            loss2_avg = []
            loss3_avg = []
            t_start = t_end

        count += 1
        if count == 25000: break

    ## dump model
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')
    torch.save(BNNeck.module.state_dict(), './res/BNNeck.pkl')

    logger.info('everything finished')


if __name__ == '__main__':
    train()
