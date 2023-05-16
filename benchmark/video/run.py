"""
[file]          run.py
[description]   
"""
#
##
import json
import time
import torch
from sklearn.model_selection import train_test_split



# from model import *
from resnet import run_resnet
from preset import preset
from load_data import load_data_y, VideoDataset


#
##
def main_0():
    #
    ##
    print(preset)
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_num_users = preset["data"]["num_users"])
    #
    data_train_pd_y, data_test_pd_y = train_test_split(data_pd_y, test_size = 0.2, shuffle = True)
    #
    data_train_set = VideoDataset(preset["path"]["data_pre_x"], data_train_pd_y, preset["task"])
    data_test_set = VideoDataset(preset["path"]["data_pre_x"], data_test_pd_y, preset["task"])

    result = run_resnet(data_train_set, data_test_set)
    
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    
    print(result)
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent = 4)


if __name__ == "__main__":
    #
    ##
    main_0()
