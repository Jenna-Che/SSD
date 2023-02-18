import numpy as np
import cv2
from dataset import iou
import torch
import matplotlib.pyplot as plt

from collections import Counter


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    image_ = image_ * 255
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)

    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    height, width, _ = image.shape
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #image1: draw ground truth bounding boxes on image1
                #From the instruction
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = ann_box[i, 0]
                dy = ann_box[i, 1]
                dw = ann_box[i, 2]
                dh = ann_box[i, 3]
                gx = pw*dx+px
                gy = ph*dy+py
                gw = pw*np.exp(dw)
                gh = ph*np.exp(dh)
                x_min = int((gx-gw/2)*width)
                y_min = int((gy-gh/2)*height)
                x_max = int((gx+gw/2)*width)
                y_max = int((gy+gh/2)*height)
                start_point = (x_min, y_min)
                end_point = (x_max, y_max)
                color = colors[j]
                thickness = 2
                image1 = cv2.rectangle(image1, start_point, end_point, color, thickness)
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                x_min_2 = int(boxs_default[i, 4]*width)
                y_min_2 = int(boxs_default[i, 5]*height)
                x_max_2 = int(boxs_default[i, 6]*width)
                y_max_2 = int(boxs_default[i, 7]*height)
                start_point_2 = (x_min_2, y_min_2)
                end_point_2 = (x_max_2, y_max_2)
                image2 = cv2.rectangle(image2, start_point_2, end_point_2, color, thickness)
                
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #From the instruction
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = pred_box[i, 0]
                dy = pred_box[i, 1]
                dw = pred_box[i, 2]
                dh = pred_box[i, 3]
                pre_x = pw * dx + px
                pre_y = ph * dy + py
                pre_w = pw * np.exp(dw)
                pre_h = ph * np.exp(dh)
                x_min_3 = int((pre_x - pre_w / 2) * width)
                y_min_3 = int((pre_y - pre_h / 2) * height)
                x_max_3 = int((pre_x + pre_w / 2) * width)
                y_max_3 = int((pre_y + pre_h / 2) * height)
                start_point = (x_min_3, y_min_3)
                end_point = (x_max_3, y_max_3)
                color = colors[j] 
                thickness = 2
                image3 = cv2.rectangle(image3, start_point, end_point, color, thickness)

                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                x_min_4 = int(boxs_default[i, 4]*width)
                y_min_4 = int(boxs_default[i, 5]*height)
                x_max_4 = int(boxs_default[i, 6]*width)
                y_max_4 = int(boxs_default[i, 7]*height)
                start_point_4 = (x_min_4, y_min_4)
                end_point_4 = (x_max_4, y_max_4)
                image4 = cv2.rectangle(image4, start_point_4, end_point_4, color, thickness)

    #combine images into one
    image = np.zeros([height*2,width*2,3], np.uint8)
    image[:height,:width] = image1
    image[:height,width:] = image2
    image[height:,:width] = image3
    image[height:,width:] = image4

    file_name = windowname + ".jpg"
    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.5):
    #non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output: dim
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    #B
    res_box = np.zeros_like(box_)
    res_conf = np.zeros_like(confidence_)

    pred_box8 = getBox8(box_, boxs_default)

    for class_num in range(0, 3):
        #Select the bounding box in A with the highest probability
        curr_c = confidence_[:, class_num]  # [540, 1]
        idx = np.argmax(curr_c)
        prob_max = curr_c[idx]
        maxb = box_[idx]  # [tx, ty, tw, th]

        #If that highest probability is greater than a threshold (threshold=0.5), proceed; otherwise, the NMS is done
        while prob_max > threshold:
            # Denote the bounding box with the highest probability as x. Move x from A to B.
            res_box[idx, :] = maxb
            res_conf[idx, :] = confidence_[idx]
            confidence_[idx, :] = 0
            box_[idx, :] = 0
            gx, gy, gw, gh, x_min, y_min, x_max, y_max = pred_box8[idx]
            ious = iou(pred_box8, x_min, y_min, x_max, y_max)  # shape = [540]

            # For all boxes in A, if a box has IOU greater than an overlap threshold
            # (overlap=0.5) with x, remove that box from A.
            del_idx = np.where(ious > overlap)[0]  # index to delete
            confidence_[del_idx, :] = 0
            box_[del_idx, :] = 0
            idx = np.argmax(curr_c)
            prob_max = curr_c[idx]
            maxb = box_[idx] 

    return res_conf, res_box

def getBox8(pred_box, boxes_default):
    box8 = np.zeros_like(boxes_default)
    dx = pred_box[:, 0]
    dy = pred_box[:, 1]
    dw = pred_box[:, 2]
    dh = pred_box[:, 3]
    px = boxes_default[:, 0]
    py = boxes_default[:, 1]
    pw = boxes_default[:, 2]
    ph = boxes_default[:, 3]
    gx = pw*dx+px
    gy = ph*dy+py
    gw = pw*np.exp(dw)
    gh = ph*np.exp(dh)
    box8[:, 0] = gx
    box8[:, 1] = gy
    box8[:, 2] = gw
    box8[:, 3] = gh
    
    box8[:, 4] = gx-gw/2
    box8[:, 5] = gy-gh/2
    box8[:, 6] = gx+gw/2
    box8[:, 7] = gy+gh/2
    return box8







