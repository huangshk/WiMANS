"""
[file]          mvit_v2.py
[description]   implement and evaluate video-based model MViT-v2
"""
#
##
import time
import torch
import numpy as np
from ptflops import get_model_complexity_info
from torchvision.models.video.mvit import mvit_v2_s, MViT_V2_S_Weights, PositionalEncoding
#
from preset import preset
from train import train, test
from load_data import VideoDataset

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- MViT-v2 ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
class MViTv2(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(MViTv2, self).__init__()
        #
        var_dim_output = var_y_shape[-1]
        #
        self.layer_mvit_v1 = mvit_v2_s(weights = MViT_V2_S_Weights.KINETICS400_V1)
        #
        var_size_temporal = var_x_shape[1]
        var_size_spatial = (var_x_shape[2], var_x_shape[3])
        #
        var_size = [int(np.ceil(var_s/var_stride)) for var_s, var_stride 
                    in zip((var_size_temporal,) + var_size_spatial, (2, 4, 4))]
        #
        self.layer_mvit_v1.pos_encoding = PositionalEncoding(embed_size = 96,
                                                        spatial_size = (var_size[1], var_size[2]),
                                                        temporal_size = var_size[0],
                                                        rel_pos_embed = True)
        #
        self.layer_linear = torch.nn.Linear(400, var_dim_output)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_mvit_v1(var_t)
        #
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
    
#
##
def run_mvit_v2(data_train_set: VideoDataset,
                data_test_set: VideoDataset,
                var_repeat: int,
                var_weight = None):
    """
    [description]
    : run video-based model MViT-v2
    [parameter]
    : data_train_set: VideoDataset, training set of video samples and labels
    : data_test_set: VideoDataset, test set of video samples and labels
    : var_repeat: int, number of repeated experiments
    : var_weight: dict, weights to initialize model
    [return]
    : result: dict, results of experiments
    : var_best_weight: dict, weights of trained model
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    var_x_shape = data_train_set.data_example_x.shape
    var_y_shape = data_train_set.data_example_y.reshape(-1).shape
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
    var_macs, var_params = get_model_complexity_info(MViTv2(var_x_shape, var_y_shape), 
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_mvit_v2 = MViTv2(var_x_shape, var_y_shape).to(device)
        #
        if var_weight is not None:  model_mvit_v2.load_state_dict(torch.load(var_weight))
        #
        optimizer = torch.optim.Adam(model_mvit_v2.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_mvit_v2, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_set = data_train_set,
                                data_test_set = data_test_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device)
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_mvit_v2.load_state_dict(var_best_weight)
        #
        result_acc, result_dict, _ = test(model_mvit_v2, 
                                          loss, 
                                          data_test_set, 
                                          preset["nn"]["threshold"], 
                                          preset["nn"]["batch_size"], device)
        #
        var_time_2 = time.time()
        #
        ## --------------------------------------- Output -----------------------------------------
        #
        result["repeat_" + str(var_r)] = result_dict
        #
        result_accuracy.append(result_acc)
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        #
        print("repeat_" + str(var_r), result_accuracy)
        print(result)
    #
    ##
    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    result["complexity"] = {"parameter": var_params, "flops": var_macs * 2}
    #
    return result, var_best_weight