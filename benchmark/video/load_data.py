"""
[file]          load_data.py
[description]   
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
class VideoDataset(torch.utils.data.Dataset):
    #
    ##
    def __init__(self,
                 var_path_data_x, 
                 var_label_list,
                 data_y):
        #
        ##
        super(VideoDataset, self).__init__()
        #
        ##
        self.var_path_data_x = var_path_data_x
        self.var_label_list = var_label_list
        self.data_y = data_y
        self.var_num_sample = len(var_label_list)

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
        ##
        var_path = os.path.join(self.var_path_data_x, self.var_label_list[var_i] + ".mp4")
        data_i_x, _, _ = torchvision.io.read_video(var_path, output_format = "TCHW")
        data_i_y = self.data_y[var_i]
        #
        return data_i_x, data_i_y
        
#
##
def load_data_y(var_path_data_y,
                var_environment = None, 
                var_num_users = None):
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
        if data_video.shape[0] != 90:
            data_error.append(var_path)
    print(data_error)
        
    

#
##
def split_train_test(var_label_list, 
                     var_test_size = 0.2, 
                     var_shuffle = True, 
                     var_random_state = None):
    #
    ##
    var_index = list(range(len(var_label_list)))
    ##
    if var_random_state is not None: random.seed(var_random_state)
    #
    if var_shuffle: random.shuffle(var_index)
    #
    var_num_test = int(var_test_size * len(var_index))
    #
    var_label_list_test = [var_label_list[var_i] for var_i in var_index[:var_num_test]]
    var_label_list_train = [var_label_list[var_i] for var_i in var_index[var_num_test:]]
    #
    return var_label_list_train, var_label_list_test

#
##
def encode_data_y(data_pd_y, var_task):
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
def encode_activity(data_pd_y, var_encoding):
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
def encode_location(data_pd_y, var_encoding):
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