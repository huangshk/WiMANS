"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##
import json
import argparse
from sklearn.model_selection import train_test_split
#
from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y

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
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    #
    ## load annotation file as labels
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_wifi_band = preset["data"]["wifi_band"], 
                            var_num_users = preset["data"]["num_users"])
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    ## load CSI amplitude
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    ## encode labels
    data_y = encode_data_y(data_pd_y, var_task)
    #
    ## a training set (80%) and a test set (20%)
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, 
                                                                            test_size = 0.2, 
                                                                            shuffle = True, 
                                                                            random_state = 39)
    #
    ## select a WiFi-based model
    if var_model == "ST-RF": run_model = run_strf
    #
    elif var_model == "MLP": run_model = run_mlp
    #
    elif var_model == "LSTM": run_model = run_lstm
    #
    elif var_model == "CNN-1D": run_model = run_cnn_1d
    #
    elif var_model == "CNN-2D": run_model = run_cnn_2d
    #
    elif var_model == "CLSTM": run_model = run_cnn_lstm
    #
    elif var_model == "ABLSTM": run_model = run_ablstm
    #
    elif var_model == "THAT": run_model = run_that
    #
    ## run WiFi-based model
    result = run_model(data_train_x, data_train_y, 
                      data_test_x, data_test_y, var_repeat)
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print(result)
    #
    ## save results
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent = 4)

#
##
if __name__ == "__main__":
    #
    ##
    run()