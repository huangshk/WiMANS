"""
[file]          preprocess.py
[description]   preprocess WiFi CSI data
"""
#
##
import os
import argparse
import numpy as np
import scipy.io as scio

#
##
def mat_to_amp(data_mat):
    """
    [description]
    : calculate amplitude of raw WiFi CSI data
    [parameter]
    : data_mat: dict, raw WiFi CSI data from *.mat files
    [return]
    : data_csi_amp: numpy array, CSI amplitude
    """
    #
    ## 
    var_length = data_mat["trace"].shape[0]
    #
    data_csi_amp = [abs(data_mat["trace"][var_t][0][0][0][-1]) for var_t in range(var_length)]
    #
    data_csi_amp = np.array(data_csi_amp, dtype = np.float32)
    #
    return data_csi_amp

#
##
def extract_csi_amp(var_dir_mat, 
                    var_dir_amp):
    """
    [description]
    : read raw WiFi CSI files (*.mat), calcuate CSI amplitude, and save amplitude (*.npy)
    [parameter]
    : var_dir_mat: string, directory to read raw WiFi CSI files (*.mat)
    : var_dir_amp: string, directory to save WiFi CSI amplitude (*.npy)
    """
    #
    ##
    var_path_mat = os.listdir(var_dir_mat)
    #
    for var_c, var_path in enumerate(var_path_mat):
        #
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        #
        data_csi_amp = mat_to_amp(data_mat)
        #
        print(var_c, data_csi_amp.shape)
        #
        var_path_save = os.path.join(var_dir_amp, var_path.replace(".mat", ".npy"))
        #
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_amp)

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
    var_args.add_argument("--dir_mat", default = "dataset/wifi_csi/mat", type = str)
    var_args.add_argument("--dir_amp", default = "dataset/wifi_csi/amp", type = str)
    #
    return var_args.parse_args()

#
##
if __name__ == "__main__":
    #
    ##
    var_args = parse_args()
    #
    extract_csi_amp(var_dir_mat = var_args.dir_mat, 
                    var_dir_amp = var_args.dir_amp)