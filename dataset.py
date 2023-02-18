import pwd
from turtle import width
from cv2 import CAP_PROP_XI_ACQ_BUFFER_SIZE, sqrt
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
import numpy as np
import os
import cv2


def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    boxes = np.zeros((135, 4, 8))
    idx, num = 0, 0
    for cell_size in layers:  # cell_size [10, 5, 3, 1]
        lsize = large_scale[idx]
        ssize = small_scale[idx]
        cell_step = 1 / cell_size
        box1 = [ssize, ssize]
        box2 = [lsize, lsize]
        box3 = [lsize * np.sqrt(2), lsize / np.sqrt(2)]
        box4 = [lsize / np.sqrt(2), lsize * np.sqrt(2)]

        # [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
        for x in range(cell_size):
            for y in range(cell_size):
                center_x = x * cell_step + cell_step / 2
                center_y = y * cell_step + cell_step / 2

                # first box
                x_min1 = center_x - box1[0] / 2
                x_max1 = center_x + box1[0] / 2
                y_min1 = center_y - box1[1] / 2
                y_max1 = center_y + box1[1] / 2
                boxes[num, 0, :] = np.clip([center_x, center_y, box1[0], box1[1], x_min1, y_min1, x_max1, y_max1], 0, 1)

                # second box
                x_min2 = center_x - box2[0] / 2
                x_max2 = center_x + box2[0] / 2
                y_min2 = center_y - box2[1] / 2
                y_max2 = center_y + box2[1] / 2
                boxes[num, 1, :] = np.clip([center_x, center_y, box2[0], box2[1], x_min2, y_min2, x_max2, y_max2], 0, 1)

                # third box
                x_min3 = center_x - box3[0] / 2
                x_max3 = center_x + box3[0] / 2
                y_min3 = center_y - box3[1] / 2
                y_max3 = center_y + box3[1] / 2
                boxes[num, 2, :] = np.clip([center_x, center_y, box3[0], box3[1], x_min3, y_min3, x_max3, y_max3], 0, 1)

                # forth box
                x_min4 = center_x - box4[0] / 2
                x_max4 = center_x + box4[0] / 2
                y_min4 = center_y - box4[1] / 2
                y_max4 = center_y + box4[1] / 2
                boxes[num, 3, :] = np.clip([center_x, center_y, box4[0], box4[1], x_min4, y_min4, x_max4, y_max4], 0, 1)
                num += 1
        idx += 1

    boxes = boxes.reshape((540, 8))
    return boxes


def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)  
    ious_true = ious>threshold

    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    #grounf box
    gx = (x_max + x_min) / 2.0
    gy = (y_max + y_min) / 2.0
    gw = x_max - x_min
    gh = y_max - y_min

    conf = [0,0,0,0]
    conf[cat_id] = 1
    indices = np.where(ious_true == True)[0]
    for i in indices:
        px = boxs_default[i,0]
        py = boxs_default[i,1]
        pw = boxs_default[i,2]
        ph = boxs_default[i,3]
        
        ann_box[i,:] = [(gx-px)/pw, 
                        (gy-py)/ph, 
                        np.log(gw/pw), 
                        np.log(gh/ph)]
        ann_confidence[i,:] = conf
    
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if len(indices) == 0:
        ious_true = np.argmax(ious)
        px = boxs_default[ious_true,0]
        py = boxs_default[ious_true,1]
        pw = boxs_default[ious_true,2]
        ph = boxs_default[ious_true,3]

        ann_box[ious_true,:] = [(gx-px)/pw, 
                                (gy-py)/ph, 
                                np.log(gw/pw), 
                                np.log(gh/ph)]
        ann_confidence[ious_true,:] = conf

    return ann_box, ann_confidence



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320, augmentation=False):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        self.augmentation = augmentation
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if self.anndir and self.train == True:
            self.img_names = self.img_names[:round(len(self.img_names)*0.9)]
        elif self.anndir and self.train == False:
            self.img_names = self.img_names[round(len(self.img_names)*0.9):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"

        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        height, width, _ = image.shape
        if self.anndir:
            class_id = []
            x_min = []
            y_min = []
            x_max = []
            y_max = []
            with open(ann_name,'r') as f:
                for line in f:
                    c_id,minmum_x,minmum_y,tw,th = line.split()
                    c_id = int(c_id)
                    minmum_x = float(minmum_x)
                    minmum_y = float(minmum_y)
                    tw = float(tw)
                    th = float(th)

                    class_id.append(c_id) 
                    x_min.append(minmum_x)  
                    y_min.append(minmum_y) 
                    x_max.append(minmum_x + tw) 
                    y_max.append(minmum_y + th) 
                    
            class_id,x_min,y_min,x_max,y_max = np.asarray(class_id), np.asarray(x_min), np.asarray(y_min), np.asarray(x_max), np.asarray(y_max)

        if self.augmentation:
            random_x_min = np.random.randint(0, np.min(x_min),size=1)[0]
            random_y_min = np.random.randint(0, np.min(y_min),size=1)[0]
            random_x_max = np.random.randint(np.max(x_max)+1,width,size=1)[0]
            random_y_max = np.random.randint(np.max(y_max)+1,height,size=1)[0]

            x_min -= random_x_min
            y_min -= random_y_min
            x_max -= random_x_min
            y_max -= random_y_min
            
            width = random_x_max - random_x_min
            height = random_y_max - random_y_min
            image = image[random_y_min:random_y_max,random_x_min:random_x_max,:]
            

        if self.anndir: #(class_id,x_min,y_min,x_max,y_max)
            for i in range(len(class_id)):
                ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,
                                               class_id[i], x_min[i]/width, y_min[i]/height,
                                               x_max[i]/width, y_max[i]/height)
        
        preprocess = transforms.Compose([transforms.ToTensor(),transforms.Resize([self.image_size,self.image_size])])
        image  = preprocess(image)
        ann_box = torch.from_numpy(ann_box)
        ann_confidence = torch.from_numpy(ann_confidence)
        
        return image, ann_box, ann_confidence