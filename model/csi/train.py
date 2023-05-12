import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
from model_rf import run_rf
from model_mlp import run_mlp
from model_lstm import run_lstm
from model_cnn_1d import run_cnn_1d
from model_cnn_2d import run_cnn_2d
from model_cnn_lstm import run_cnn_lstm
from model_ablstm import run_ablstm

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
                            var_wifi_band = preset["data"]["wifi_band"], 
                            var_num_users = preset["data"]["num_users"])
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    data_y = encode_data_y(data_pd_y, preset["task"])

    #
    ##
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, 
                                                                            test_size = 0.2, 
                                                                            shuffle = True, 
                                                                            random_state = 39)
    #
    # result = run_rf(data_train_x, data_train_y, data_test_x, data_test_y)
    # result = run_mlp(data_train_x, data_train_y, data_test_x, data_test_y)
    # result = run_lstm(data_train_x, data_train_y, data_test_x, data_test_y)
    # result = run_cnn_1d(data_train_x, data_train_y, data_test_x, data_test_y)
    # result = run_cnn_2d(data_train_x, data_train_y, data_test_x, data_test_y)
    # result = run_cnn_lstm(data_train_x, data_train_y, data_test_x, data_test_y)
    result = run_ablstm(data_train_x, data_train_y, data_test_x, data_test_y)
    #
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print(result)
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent = 4)

#
##
if __name__ == "__main__":
    #
    ##
    # main_0()

    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["classroom"]
    preset["data"]["wifi_band"] = ["2.4"]
    preset["path"]["save"] = "result_activity_ablstm_classroom_24.json"
    main_0()
    
    #
    preset["task"] = "activity"
    preset["data"]["environment"] = ["classroom"]
    preset["data"]["wifi_band"] = ["5"]
    preset["path"]["save"] = "result_activity_ablstm_classroom_5.json"
    main_0()
    
    #
    preset["task"] = "activity"
    preset["data"]["environment"] = ["classroom"]
    preset["data"]["wifi_band"] = ["2.4", "5"]
    preset["path"]["save"] = "result_activity_ablstm_classroom_245.json"
    main_0()

    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["meeting_room"]
    preset["data"]["wifi_band"] = ["2.4"]
    preset["path"]["save"] = "result_activity_ablstm_meeting_24.json"
    main_0()
    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["meeting_room"]
    preset["data"]["wifi_band"] = ["5"]
    preset["path"]["save"] = "result_activity_ablstm_meeting_5.json"
    main_0()
    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["meeting_room"]
    preset["data"]["wifi_band"] = ["2.4", "5"]
    preset["path"]["save"] = "result_activity_ablstm_meeting_245.json"
    main_0()

    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["empty_room"]
    preset["data"]["wifi_band"] = ["2.4"]
    preset["path"]["save"] = "result_activity_ablstm_empty_24.json"
    main_0()
    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["empty_room"]
    preset["data"]["wifi_band"] = ["5"]
    preset["path"]["save"] = "result_activity_ablstm_empty_5.json"
    main_0()
    #
    ##
    preset["task"] = "activity"
    preset["data"]["environment"] = ["empty_room"]
    preset["data"]["wifi_band"] = ["2.4", "5"]
    preset["path"]["save"] = "result_activity_ablstm_empty_245.json"
    main_0()

