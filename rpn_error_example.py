#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from "frcnn-from-scratch-with-keras.simple_parser import get_data

from keras_frcnn.simple_parser import get_data
from keras_frcnn import data_generators
from keras_frcnn import config
from keras_frcnn import vgg as nn
import keras_frcnn.roi_helpers as roi_helpers

import numpy as np

import cv2 as cv2
import matplotlib.pyplot as plt

def rpn_to_roi_FIXED(y_rpn_cls,y_rpn_regr,C,max_boxes=300,overlap_thresh=0.9):
    #outputs (x1,y1,x2,y2)
    assert y_rpn_cls.shape[0] == 1

    y_rpn_regr = y_rpn_regr / C.std_scaling
    (output_height,output_width) = y_rpn_cls.shape[1:3]
    
    boxes = np.zeros((output_height,output_width,y_rpn_cls.shape[3],4))
    anchor_idx = 0
    for anchor_size_idx in range(len(C.anchor_box_scales)):
        for anchor_ratio_idx in range(len(C.anchor_box_ratios)):
            
            #define anchor
            anchor_x = C.anchor_box_scales[anchor_size_idx] * C.anchor_box_ratios[anchor_ratio_idx][0]
            anchor_y = C.anchor_box_scales[anchor_size_idx] * C.anchor_box_ratios[anchor_ratio_idx][1]
            
            for i in range(output_width):
                x1_anch = C.rpn_stride * (i + 0.5) - anchor_x / 2
                x2_anch = C.rpn_stride * (i + 0.5) + anchor_x / 2
                
                if x1_anch < 0 or x2_anch > output_width*C.rpn_stride:
                    continue
                
                for j in range(output_height):
                    y1_anch = C.rpn_stride * (j + 0.5) - anchor_y / 2
                    y2_anch = C.rpn_stride * (j + 0.5) + anchor_y /2
                    
                    if y1_anch < 0 or y2_anch > output_height*C.rpn_stride:
                        continue
                    
                    tx = y_rpn_regr[:,j,i,4*anchor_idx + 0]
                    ty = y_rpn_regr[:,j,i,4*anchor_idx + 1]
                    tw = np.exp(y_rpn_regr[:,j,i,4*anchor_idx + 2])
                    th = np.exp(y_rpn_regr[:,j,i,4*anchor_idx + 3])
                    
                    twa = (x2_anch - x1_anch)
                    cxa = (x2_anch + x1_anch)/2.0
                    
                    tha = (y2_anch - y1_anch)
                    cya = (y2_anch + y1_anch)/2.0
                    
                    x1 = -0.5*twa*tw + tx*twa + cxa
                    x2 = x1 + tw*twa
                    y1 = -0.5*tha*th + ty*tha + cya
                    y2 = y1 + th*tha

                    boxes[j,i,anchor_idx,:] = [x1,y1,x2,y2]
                    
            anchor_idx += 1
                    
    boxes = boxes.reshape((-1,4))
    probs = y_rpn_cls.reshape((-1,))

    idxs = np.where(np.sum(boxes,axis=1)!=0)
    boxes = boxes[idxs]
    probs = probs[idxs]
    
    boxes,probs = roi_helpers.non_max_suppression_fast(boxes, probs)
    boxes = boxes / 16
    return boxes


C = config.Config()


all_imgs, classes_count, class_mapping = get_data("test.txt")

data_gen = data_generators.get_anchor_gt(all_imgs, classes_count, C, nn.get_img_output_length, 'tf', mode='train')

X, [y1,y2], img_data = next(data_gen)

rois = roi_helpers.rpn_to_roi(np.copy(y1[:,:,:,y1.shape[3]//2:]),
                              np.copy(y2[:,:,:,y2.shape[3]//2:]),
                              C,'tf')

rois_fixed = rpn_to_roi_FIXED(np.copy(y1[:,:,:,y1.shape[3]//2:]),
                              np.copy(y2[:,:,:,y2.shape[3]//2:]),
                              C)

x1 = (((all_imgs[0])['bboxes'])[0])['x1'] // 2 
y1 = (((all_imgs[0])['bboxes'])[0])['y1'] // 2
x2 = (((all_imgs[0])['bboxes'])[0])['x2'] // 2
y2 = (((all_imgs[0])['bboxes'])[0])['y2'] // 2

image = cv2.imread("test.jpg")
image = cv2.resize(image, (400,300))
image2 = np.copy(image)

for i in range(1):
    pnt1 = (int(rois[i,0]*C.rpn_stride),int(rois[i,1]*C.rpn_stride))
    pnt2 = (int(rois[i,2]*C.rpn_stride),int(rois[i,3]*C.rpn_stride))
    cv2.rectangle(image,pnt1,pnt2,color=(255,0,0),thickness=8)
    
for i in range(1):
    pnt1 = (int(rois_fixed[i,0]*C.rpn_stride),int(rois_fixed[i,1]*C.rpn_stride))
    pnt2 = (int(rois_fixed[i,2]*C.rpn_stride),int(rois_fixed[i,3]*C.rpn_stride))
    cv2.rectangle(image2,pnt1,pnt2,color=(255,0,0),thickness=8)

cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,255,0),thickness=2)
cv2.rectangle(image2,(x1,y1),(x2,y2),color=(0,255,0),thickness=2)


plt.subplot(121)
plt.title("initial implementation")
plt.imshow(image)
plt.subplot(122)
plt.title("fixed implementaion")
plt.imshow(image2)
