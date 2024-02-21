"""
[file]          load_data.py
[description]   load annotation file and video files, and encode labels
"""
#
##
import os
import random
import numpy as np
import pandas as pd
import torch, torchvision
#
from preset import preset

#
##
## ------------------------------------------------------------------------------------------ ##
## ------------------------------------ VideoDataset ---------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class VideoDataset(torch.utils.data.Dataset):
    #
    ##
    def __init__(self,
                 var_path_pre_x,
                 data_pd_y,
                 var_task,
                 var_frame_stride):
        #
        ##
        super(VideoDataset, self).__init__()
        #
        ##
        var_label_list = data_pd_y["label"].to_list()
        self.var_path_list = [os.path.join(var_path_pre_x, var_label + ".npy") 
                              for var_label in var_label_list]
        #
        self.data_y = encode_data_y(data_pd_y, var_task)
        #
        self.var_num_sample = len(var_label_list)
        #
        self.var_frame_stride = var_frame_stride
        #
        self.data_example_x = np.swapaxes(np.load(self.var_path_list[0])[::var_frame_stride], 1, 0)
        self.data_example_y = self.data_y[0]

    #
    ##
    def __len__(self):
        #
        ##
        return self.var_num_sample
    
    #
    ##
    def __getitem__(self, var_i):
        #
        ## "TCHW" -> "CTHW"
        data_i_x = np.swapaxes(np.load(self.var_path_list[var_i])[::self.var_frame_stride], 1, 0)
        data_i_y = self.data_y[var_i]
        #
        return data_i_x, data_i_y
        
#
##
def load_data_y(var_path_data_y,
                var_environment = None, 
                var_num_users = None):
    """
    [description]
    : load annotation file (*.csv) as a pandas dataframe
    : according to selected environment(s), and number(s) of users
    [parameter]
    : var_path_data_y: string, path of annotation file
    : var_environment: list, selected environment(s), e.g., ["classroom"]
    : var_num_users: list, selected number(s) of users, e.g., ["0", "1", "2"]
    [return]
    : data_pd_y: pandas dataframe, labels of selected data
    """
    #
    ##
    data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
    #
    if var_environment is not None:
        data_pd_y = data_pd_y[data_pd_y["environment"].isin(var_environment)]
    #
    if var_num_users is not None:
        data_pd_y = data_pd_y[data_pd_y["number_of_users"].isin(var_num_users)]
    #
    return data_pd_y

#
##
def load_data_x(var_path_data_x, 
                var_label_list):
    """
    [description]
    : load and check video files (*.mp4)
    : according to a label list of selected data
    [parameter]
    : var_path_data_x: string, directory of video files
    : var_label_list: list, selected labels
    """
    #
    ##
    var_path_list = [os.path.join(var_path_data_x, var_label + ".mp4") for var_label in var_label_list]
    #
    data_error = []
    #
    for var_path in var_path_list:
        #
        data_video, _, _ = torchvision.io.read_video(var_path, output_format = "TCHW")
        #
        print(var_path, data_video.shape, data_video.shape[0] == 90)
        #
        if data_video.shape[0] != 90: data_error.append(var_path)
    #
    print(data_error)
        
#
##
def encode_data_y(data_pd_y, 
                  var_task):
    """
    [description]
    : encode labels according to specific task
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_task: string, indicate task
    [return]
    : data_y: numpy array, label encoding of task
    """
    #
    ##
    if var_task == "identity":
        #
        data_y = encode_identity(data_pd_y)
    #
    elif var_task == "activity":
        #
        data_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    elif var_task == "location":
        #
        data_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    return data_y

#
##
def encode_identity(data_pd_y):
    """
    [description]
    : encode identity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    [return]
    : data_identity_onehot_y: numpy array, onehot encoding for identity labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    # 
    data_identity_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_identity_y[data_identity_y != "nan"] = 1
    data_identity_y[data_identity_y == "nan"] = 0
    #
    data_identity_onehot_y = data_identity_y.astype("int8")
    #
    return data_identity_onehot_y

#
##
def encode_activity(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode activity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different activities
    [return]
    : data_activity_onehot_y: numpy array, onehot encoding for activity labels
    """
    #
    ##
    data_activity_pd_y = data_pd_y[["user_1_activity", "user_2_activity", 
                                    "user_3_activity", "user_4_activity", 
                                    "user_5_activity", "user_6_activity"]]
    #
    data_activity_y = data_activity_pd_y.to_numpy(copy = True).astype(str)
    #
    data_activity_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_activity_y])
    #
    return data_activity_onehot_y

#
##
def encode_location(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode location labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different locations
    [return]
    : data_location_onehot_y: numpy array, onehot encoding for location labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    #
    data_location_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_location_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_location_y])
    #
    return data_location_onehot_y