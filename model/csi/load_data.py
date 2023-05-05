import os
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
#
from preset import preset

#
##
def load_data_y(var_path_data_y,
                var_environment = None, 
                var_wifi_band = None, 
                var_num_users = None):
    #
    ##
    data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
    #
    if var_environment is not None:
        data_pd_y = data_pd_y[data_pd_y["environment"].isin(var_environment)]
    #
    if var_wifi_band is not None:
        data_pd_y = data_pd_y[data_pd_y["wifi_band"].isin(var_wifi_band)]
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
    var_path_list = [os.path.join(var_path_data_x, var_label + ".npy") for var_label in var_label_list]
    #
    data_x = []
    #
    for var_path in var_path_list:
        #
        data_csi = np.load(var_path)
        #
        var_pad_length = preset["data"]["length"] - data_csi.shape[0]
        #
        data_csi_pad = np.pad(data_csi, ((var_pad_length, 0), (0, 0), (0, 0), (0, 0)))
        #
        data_x.append(data_csi_pad)
    #
    data_x = np.array(data_x)
    #
    return data_x

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

#
##
def test_load_data_y():
    #
    ##
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["classroom"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"], 
                      var_num_users = ["1", "2", "3"]).describe())

#
##
def test_load_data_x():
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = ["meeting_room"], 
                            var_wifi_band = ["2.4"], 
                            var_num_users = None)
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    print(data_x.shape)

#
##
def test_encode_identity():
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_identity_onehot_y = encode_identity(data_pd_y)
    #
    print(data_identity_onehot_y.shape)
    #
    print(data_identity_onehot_y[2000])

#
##
def test_encode_activity():
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_activity_onehot_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    print(data_activity_onehot_y.shape)
    #
    print(data_activity_onehot_y[1560])

#
##
def test_encode_location():
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_location_onehot_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    print(data_location_onehot_y.shape)
    #
    print(data_location_onehot_y[1560])

#
##
if __name__ == "__main__":
    #
    ##
    test_load_data_y()
    #
    test_load_data_x()
    #
    test_encode_identity()
    #
    test_encode_activity()
    #
    test_encode_location()