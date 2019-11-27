from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import json
import numpy

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_folder", type=str, default="test_image", help="path to dataset")
#     parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
#     # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
#     # parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
#     parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
#     parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
#     parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
#     parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
#     parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
#     parser.add_argument("--checkpoint_model", type=str, default="checkpoints/yolov3_ckpt_50.pth", help="path to checkpoint model")
#     opt = parser.parse_args()
#     print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4
    # Set up model
    model = Darknet("config/yolov3-custom.cfg", img_size).to(device)

    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_50.pth"))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder('test_image', img_size),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    total_time = 0
    prev_time = time.time()
	
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        # print("imgshape:", input_imgs.shape)
        input_imgs = Variable(input_imgs.type(Tensor))
        # print("imghall:", input_imgs.shape)
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            # print("detections", detections)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
            # print("detectionsafter:", detections)
        # Log progress
        current_time = time.time()
        inference_time = current_time - prev_time
        prev_time = current_time
        total_time = total_time + inference_time
        print("\t+ Batch %d, Inference Time: %.3f ms" % (batch_i, inference_time*1000))
	
print("Avg of Inference Time: %.3f ms" % (total_time/(batch_i+1)*1000))	
