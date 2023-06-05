"""
[file]          preprocess.py
[description]   preprocess video data
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch, torchvision
#
from torchvision.models.video import R3D_18_Weights, S3D_Weights, MViT_V1_B_Weights, MViT_V2_S_Weights, Swin3D_T_Weights, Swin3D_S_Weights
from preset import preset

#
##
def preprocess_video(var_path_data_x, 
                     var_path_data_y,
                     var_model,
                     var_path_data_pre_x):
    """
    [description]
    : preprocess video data for video-based models
    [parameter]
    : var_path_data_x: string, directory to read raw video files (*.mp4)
    : var_path_data_y: string, path of annotation file
    : var_model: string, model for which videos should be preprocessed
    : var_path_data_pre_x: string, directory to save preprocessed videos (*.npy)
    """
    #
    ##
    if var_model == "ResNet":
        transform = R3D_18_Weights.DEFAULT.transforms()
    #
    elif var_model == "S3D":
        transform = S3D_Weights.DEFAULT.transforms()
    #
    elif var_model == "MViT-v1":
        transform = MViT_V1_B_Weights.DEFAULT.transforms()
    #
    elif var_model == "MViT-v2":
        transform = MViT_V2_S_Weights.DEFAULT.transforms()
    #
    elif var_model == "Swin-T":
        transform = Swin3D_T_Weights.DEFAULT.transforms()
    #
    elif var_model == "Swin-S":
        transform = Swin3D_S_Weights.DEFAULT.transforms()
    #
    ##
    data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    print(len(var_label_list))
    #
    for var_i, var_label in enumerate(var_label_list):
        #
        var_path = os.path.join(var_path_data_x, var_label + ".mp4")
        #
        data_video_x, _, _ = torchvision.io.read_video(var_path, output_format = "TCHW")
        #
        if data_video_x.shape[0] != 90: print(var_label, "Warning!")
        #
        data_pre_x = transform(data_video_x)
        #
        data_pre_x = torch.permute(data_pre_x, (1, 0, 2, 3))  # TCHW
        #
        print(var_label, data_video_x.shape, data_pre_x.shape)
        #
        np.save(os.path.join(var_path_data_pre_x, var_label + ".npy"), data_pre_x)
    
#
##
def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--path_data_x", default = preset["path"]["data_x"], type = str)
    var_args.add_argument("--path_data_y", default = preset["path"]["data_y"], type = str)
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--path_data_pre_x", default = preset["path"]["data_pre_x"], type = str)
    #
    return var_args.parse_args()

#
##
if __name__ == "__main__":
    #
    ##
    var_args = parse_args()
    #
    var_time = time.time()
    #
    preprocess_video(var_path_data_x = var_args.path_data_x, 
                     var_path_data_y = var_args.path_data_y,
                     var_model = var_args.model,
                     var_path_data_pre_x = var_args.path_data_pre_x)
    #
    print("Preprocess Time:", time.time() - var_time)
