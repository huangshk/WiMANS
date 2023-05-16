import os
import time
import numpy as np
import pandas as pd
from preset import preset
import torch
import torchvision
from torchvision.models.video import R3D_18_Weights, MViT_V1_B_Weights, S3D_Weights, MViT_V2_S_Weights, Swin3D_T_Weights

#
##
def preprocess_video(var_path_data_x, 
                     var_path_data_y,
                     var_target,
                     var_path_save):
    #
    ##
    if var_target == "ResNet":
        transform = R3D_18_Weights.DEFAULT.transforms()
    #
    elif var_target == "MViTv1":
        transform = MViT_V1_B_Weights.DEFAULT.transforms()
    #
    elif var_target == "S3D":
        transform = S3D_Weights.DEFAULT.transforms()
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
        np.save(os.path.join(var_path_save, var_label + ".npy"), data_pre_x)
        #
        # if var_i >=20: break
    


#
##
if __name__ == "__main__":
    #
    var_time = time.time()
    preprocess_video(preset["path"]["data_x"], 
                     preset["path"]["data_y"], 
                     "MViTv1",
                     "/home/hwang/Lab/Project/WiMans/cache/mvit")
    print("Preprocess Time:", time.time() - var_time)
    #
    # var_time = time.time()
    # data = np.load("/home/hwang/Lab/Project/WiMans/cache/mvit/act_1_9.npy")
    # print(data.shape, data.dtype)
    # print(time.time()-var_time)