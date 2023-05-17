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

from model import *
from preset import preset
from load_data import load_data_y, VideoDataset


#
##
def main_0(var_pretrain = True):
    #
    ##
    print(preset)
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_num_users = preset["data"]["num_users"])
    #
    data_train_pd_y, data_test_pd_y = train_test_split(data_pd_y, 
                                                       test_size = 0.2, 
                                                       shuffle = True, 
                                                       random_state = 39)
    #
    data_train_set = VideoDataset(preset["path"]["data_pre_x"], data_train_pd_y, preset["task"], preset["nn"]["frame_stride"])
    data_test_set = VideoDataset(preset["path"]["data_pre_x"], data_test_pd_y, preset["task"], preset["nn"]["frame_stride"])
    #
    ##
    # run_model = run_resnet
    run_model = run_s3d
    #
    ##
    if var_pretrain:
        #
        result, var_weight = run_model(data_train_set, 
                                       data_test_set,
                                       var_repeat = preset["nn"]["repeat"])
        #
        result["data"] = preset["data"]
        result["nn"] = preset["nn"]
        #
        print(result)
        #
        var_file = open(preset["path"]["save_result"], 'w')
        json.dump(result, var_file, indent = 4)
        #
        torch.save(var_weight, preset["path"]["save_model"])
    #
    ##
    else:
        #
        result, _ = run_model(data_train_set, 
                              data_test_set,
                              var_repeat = preset["nn"]["repeat"],
                              var_weight = preset["path"]["save_model"])
        #
        result["data"] = preset["data"]
        result["nn"] = preset["nn"]
        #
        print(result)
        #
        var_file = open(preset["path"]["save_result"], 'w')
        json.dump(result, var_file, indent = 4)


if __name__ == "__main__":
    #
    ##
    # main_0()
    #
    preset["task"] = "identity"
    preset["nn"]["repeat"] = 1
    preset["nn"]["epoch"] = 10
    preset["nn"]["lr"] = 1e-3
    preset["data"]["environment"] = ["classroom"]
    preset["path"]["save_result"] = "result_identity_s3d_classroom_pre.json"
    preset["path"]["save_model"] = "model_identity_s3d_classroom.pt"
    main_0(var_pretrain = True)
    # #
    preset["nn"]["repeat"] = 10
    preset["nn"]["epoch"] = 1
    preset["nn"]["lr"] = 1e-4
    preset["path"]["save_result"] = "result_identity_s3d_classroom.json"
    main_0(var_pretrain = False)


    #
    preset["task"] = "location"
    preset["nn"]["repeat"] = 1
    preset["nn"]["epoch"] = 10
    preset["nn"]["lr"] = 1e-3
    preset["data"]["environment"] = ["classroom"]
    preset["path"]["save_result"] = "result_location_s3d_classroom_pre.json"
    preset["path"]["save_model"] = "model_location_s3d_classroom.pt"
    main_0(var_pretrain = True)
    # #
    preset["nn"]["repeat"] = 10
    preset["nn"]["epoch"] = 1
    preset["nn"]["lr"] = 1e-4
    preset["path"]["save_result"] = "result_location_s3d_classroom.json"
    main_0(var_pretrain = False)

    #
    preset["task"] = "activity"
    preset["nn"]["repeat"] = 1
    preset["nn"]["epoch"] = 10
    preset["nn"]["lr"] = 1e-3
    preset["data"]["environment"] = ["classroom"]
    preset["path"]["save_result"] = "result_activity_s3d_classroom_pre.json"
    preset["path"]["save_model"] = "model_activity_s3d_classroom.pt"
    main_0(var_pretrain = True)
    # #
    preset["nn"]["repeat"] = 10
    preset["nn"]["epoch"] = 1
    preset["nn"]["lr"] = 1e-4
    preset["path"]["save_result"] = "result_activity_s3d_classroom.json"
    main_0(var_pretrain = False)

