import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
     # loss -- a single number for the value of the loss function, [1]
    
       
    _,_,num_of_classes = pred_confidence.shape
    ann_box = ann_box.reshape((-1,4))
    pred_box = pred_box.reshape((-1,4))
    pred_confidence = pred_confidence.reshape((-1,num_of_classes))
    ann_confidence = ann_confidence.reshape((-1,num_of_classes))
    
    ind_obj = torch.where(ann_confidence[:,-1]==0)[0]
    ind_nonobj = torch.where(ann_confidence[:,-1]!=0)[0]

    loss_cls_obj = F.cross_entropy(pred_confidence[ind_obj],ann_confidence[ind_obj])
    loss_cls_nonobj = F.cross_entropy(pred_confidence[ind_nonobj],ann_confidence[ind_nonobj])
    loss_cls = loss_cls_obj + 3*loss_cls_nonobj

    loss_box = F.smooth_l1_loss(pred_box[ind_obj],ann_box[ind_obj])# ,reduction='sum') #/ num_of_boxes

    loss = loss_cls + loss_box

    return loss




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left_red1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 100]
        self.right_blue1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left_red2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 25]
        self.right_blue2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
     
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    
        self.left_red3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 9]
        self.right_blue3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)

        
        self.left_red4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)  # reshape [N, 16, 1]
        self.right_blue4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x_left_red1 = self.left_red1(x1)
        x_left_red1 = x_left_red1.reshape((-1, 16, 100))
        x_right_blue1 = self.right_blue1(x1)
        x_right_blue1 = x_right_blue1.reshape((-1, 16, 100))

        x3 = self.conv3(x2)
        x_left_red2 = self.left_red2(x2)
        x_left_red2 = x_left_red2.reshape((-1, 16, 25))
        x_right_blue2 = self.right_blue2(x2)
        x_right_blue2 = x_right_blue2.reshape((-1, 16, 25))

        x4 = self.conv4(x3)
        x_left_red3 = self.left_red3(x3)
        x_left_red3 = x_left_red3.reshape((-1, 16, 9))
        x_right_blue3 = self.right_blue3(x3)
        x_right_blue3 = x_right_blue3.reshape((-1, 16, 9))

        x_left_red4 = self.left_red4(x4)
        x_left_red4 = x_left_red4.reshape((-1, 16, 1))
        x_right_blue4 = self.right_blue4(x4)
        x_right_blue4 = x_right_blue4.reshape((-1, 16, 1))

        x_bbox = torch.cat((x_left_red1, x_left_red2, x_left_red3, x_left_red4), dim=2)  # [N, 16, 135]
        x_bbox = x_bbox.permute((0, 2, 1))  # [N, 135, 16]
        bboxes = x_bbox.reshape((-1, 540, 4))

        x_conf = torch.cat((x_right_blue1, x_right_blue2, x_right_blue3, x_right_blue4), dim=2)  # [N, 16, 135]
        x_conf = x_conf.permute((0, 2, 1))  # [N, 135, 16]
        x_conf = x_conf.reshape((-1, 540, 4))
        confidence = torch.softmax(x_conf, dim=2)

        return confidence, bboxes










