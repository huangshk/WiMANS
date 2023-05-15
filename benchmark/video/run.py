"""
[file]          run.py
[description]   
"""
#
##
import json
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from torchvision.models.video import r3d_18, R3D_18_Weights


preprocess = R3D_18_Weights.DEFAULT.transforms()

# from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y, split_train_test, VideoDataset


#
##
def main_0():
    #
    ##
    print(preset)


if __name__ == "__main__":
    #
    ##
    main_0()
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_num_users = preset["data"]["num_users"])
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    var_label_list_train, var_label_list_test = split_train_test(var_label_list)
    #
    data_pd_train_y = data_pd_y[data_pd_y["label"].isin(var_label_list_train)]
    data_pd_test_y = data_pd_y[data_pd_y["label"].isin(var_label_list_test)]
    #
    data_train_y = encode_data_y(data_pd_train_y, preset["task"])
    data_test_y = encode_data_y(data_pd_test_y, preset["task"])
    #
    data_train_set = VideoDataset(preset["path"]["data_x"], var_label_list_train, data_train_y)
    data_test_set = VideoDataset(preset["path"]["data_x"], var_label_list_test, data_test_y)
    #
    data_train_loader = torch.utils.data.DataLoader(dataset = data_train_set,
                                                    batch_size = preset["nn"]["batch_size"],
                                                    shuffle = True, num_workers = 4)

    var_time = time.time()
    #
    for data_batch in data_train_loader:
        #
        ##
        data_batch_x, data_batch_y = data_batch
        #
        data_preprocess_x = preprocess(data_batch_x)
        #
        print(time.time() - var_time, data_batch_x.shape, data_batch_y.shape, data_preprocess_x.shape)
        #
        var_time = time.time()



    #
    # data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    # print(data_x.shape)