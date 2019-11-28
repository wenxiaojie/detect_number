from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4
    # Set up model
    model = Darknet("config/yolov3-custom.cfg", img_size).to(device)
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
        print("\t+ Batch %d, Inference Time: %.3f ms" % (batch_i, inference_time * 1000))

    print("Avg of Inference Time: %.3f ms" % (total_time / (batch_i + 1) * 1000))	
