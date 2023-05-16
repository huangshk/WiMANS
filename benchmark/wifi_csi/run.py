"""
[file]          run.py
[description]   
"""
#
##
import json

from sklearn.model_selection import train_test_split

from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y


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
    # result = run_strf(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    result = run_mlp(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_lstm(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_cnn_1d(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_cnn_2d(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_cnn_lstm(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_ablstm(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
    # result = run_that(data_train_x, data_train_y, data_test_x, data_test_y, preset["nn"]["repeat"])
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
    main_0()

    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["classroom"]
    # preset["data"]["wifi_band"] = ["2.4"]
    # preset["path"]["save"] = "result_activity_that_classroom_24.json"
    # main_0()
    
    #
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["classroom"]
    # preset["data"]["wifi_band"] = ["5"]
    # preset["path"]["save"] = "result_activity_that_classroom_5.json"
    # main_0()
    
    #
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["classroom"]
    # preset["data"]["wifi_band"] = ["2.4", "5"]
    # preset["path"]["save"] = "result_activity_that_classroom_245.json"
    # main_0()

    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["meeting_room"]
    # preset["data"]["wifi_band"] = ["2.4"]
    # preset["path"]["save"] = "result_activity_that_meeting_24.json"
    # main_0()
    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["meeting_room"]
    # preset["data"]["wifi_band"] = ["5"]
    # preset["path"]["save"] = "result_activity_that_meeting_5.json"
    # main_0()
    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["meeting_room"]
    # preset["data"]["wifi_band"] = ["2.4", "5"]
    # preset["path"]["save"] = "result_activity_that_meeting_245.json"
    # main_0()

    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["empty_room"]
    # preset["data"]["wifi_band"] = ["2.4"]
    # preset["path"]["save"] = "result_activity_that_empty_24.json"
    # main_0()
    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["empty_room"]
    # preset["data"]["wifi_band"] = ["5"]
    # preset["path"]["save"] = "result_activity_that_empty_5.json"
    # main_0()
    #
    ##
    # preset["task"] = "activity"
    # preset["data"]["environment"] = ["empty_room"]
    # preset["data"]["wifi_band"] = ["2.4", "5"]
    # preset["path"]["save"] = "result_activity_that_empty_245.json"
    # main_0()

