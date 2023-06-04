"""
[file]          run.py
[description]   run video-based models
"""
#
##
import json
import argparse
import torch
from sklearn.model_selection import train_test_split
#
from model import *
from preset import preset
from load_data import load_data_y, VideoDataset
#
torch.backends.cudnn.benchmark = True

#
##
def parse_args():
    """
    [description]
    : parse the arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    #
    return var_args.parse_args()

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    #
    ## parse the arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    #
    ## load the annotation file as labels
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_num_users = preset["data"]["num_users"])
    #
    ## a training set (80%) and a test set (20%)
    data_train_pd_y, data_test_pd_y = train_test_split(data_pd_y, 
                                                       test_size = 0.2, 
                                                       shuffle = True, 
                                                       random_state = 39)
    #
    data_train_set = VideoDataset(preset["path"]["data_pre_x"], 
                                  data_train_pd_y, 
                                  var_task, 
                                  preset["nn"]["frame_stride"])
    data_test_set = VideoDataset(preset["path"]["data_pre_x"], 
                                 data_test_pd_y, 
                                 var_task, 
                                 preset["nn"]["frame_stride"])
    #
    ## select a WiFi-based model
    if var_model == "ResNet": run_model = run_resnet
    #
    elif var_model == "S3D": run_model = run_s3d
    #
    elif var_model == "MViT-v1": run_model = run_mvit_v1
    #
    elif var_model == "MViT-v2": run_model = run_mvit_v2
    #
    elif var_model == "Swin-T": run_model = run_swin_t
    #
    elif var_model == "Swin-S": run_model = run_swin_s
    #
    ## run the WiFi-based model
    result, var_weight = run_model(data_train_set, data_test_set, var_repeat,
                                   var_weight = preset["path"]["save_model"])
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    ## save results
    var_file = open(preset["path"]["save_result"], 'w')
    json.dump(result, var_file, indent = 4)
    #
    ## save model
    torch.save(var_weight, preset["path"]["save_model"])

# #
# ##
# def main_0(var_pretrain = True):
#     #
#     ##
#     print(preset)
#     #
#     ##
#     data_pd_y = load_data_y(preset["path"]["data_y"],
#                             var_environment = preset["data"]["environment"], 
#                             var_num_users = preset["data"]["num_users"])
#     #
#     data_train_pd_y, data_test_pd_y = train_test_split(data_pd_y, 
#                                                        test_size = 0.2, 
#                                                        shuffle = True, 
#                                                        random_state = 39)
#     #
#     data_train_set = VideoDataset(preset["path"]["data_pre_x"], data_train_pd_y, preset["task"], preset["nn"]["frame_stride"])
#     data_test_set = VideoDataset(preset["path"]["data_pre_x"], data_test_pd_y, preset["task"], preset["nn"]["frame_stride"])
#     #
#     ##
#     # run_model = run_resnet
#     # run_model = run_s3d
#     # run_model = run_mvit_v1
#     # run_model = run_mvit_v2
#     # run_model = run_swin_t
#     run_model = run_swin_s
#     #
#     ##
#     if var_pretrain:
#         #
#         result, var_weight = run_model(data_train_set, 
#                                        data_test_set,
#                                        var_repeat = preset["nn"]["repeat"])
#         #
#         result["data"] = preset["data"]
#         result["nn"] = preset["nn"]
#         #
#         print(result)
#         #
#         var_file = open(preset["path"]["save_result"], 'w')
#         json.dump(result, var_file, indent = 4)
#         #
#         torch.save(var_weight, preset["path"]["save_model"])
#     #
#     ##
#     else:
#         #
#         result, _ = run_model(data_train_set, 
#                               data_test_set,
#                               var_repeat = preset["nn"]["repeat"],
#                               var_weight = preset["path"]["save_model"])
#         #
#         result["data"] = preset["data"]
#         result["nn"] = preset["nn"]
#         #
#         print(result)
#         #
#         var_file = open(preset["path"]["save_result"], 'w')
#         json.dump(result, var_file, indent = 4)


if __name__ == "__main__":
    #
    ##
    run()