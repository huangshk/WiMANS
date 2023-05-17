import time
import torch
import numpy as np
#
from torchvision.models.video.mvit import mvit_v1_b, PositionalEncoding
#
from ptflops import get_model_complexity_info
#
from preset import preset
from train import train, test

#
##
def run_mvit_v1(data_train_set,
                data_test_set,
                var_repeat,
                var_weight = None):
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    var_x_shape = data_train_set[0][0].numpy().shape
    var_y_shape = data_train_set[0][1].numpy().reshape(-1).shape
    print(var_x_shape, var_y_shape)
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    #
    ##
    MViTv1 = mvit_v1_b(num_classes = var_y_shape[-1])
    input_size = [size // stride for size, stride in zip((90,) + (224, 224), (2, 4, 4))]

    print(input_size)

    MViTv1.pos_encoding = PositionalEncoding(
        embed_size = 96,
        spatial_size = (input_size[1], input_size[2]),
        temporal_size = input_size[0],
        rel_pos_embed = False
    )



    var_macs, var_params = get_model_complexity_info(MViTv1, 
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)