import time
import torch
import numpy as np
from ptflops import get_model_complexity_info
from torchvision.models.video.s3d import s3d, S3D_Weights
#
##
from preset import preset
from train import train, test
from load_data import VideoDataset

#
##
class S3DM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_y_shape):
        #
        ##
        super(S3DM, self).__init__()
        #
        var_dim_output = var_y_shape[-1]
        #
        self.layer_s3d = s3d(weights = S3D_Weights.KINETICS400_V1)
        #
        self.layer_linear = torch.nn.Linear(400, var_dim_output)

    #
    ##
    def forward(self,
                var_input):
        #
        var_t = var_input
        #
        var_t = self.layer_s3d(var_t)
        #
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output

#
##
def run_s3d(data_train_set: VideoDataset,
            data_test_set: VideoDataset,
            var_repeat: int,
            var_weight = None):
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
    var_macs, var_params = get_model_complexity_info(S3DM(var_y_shape), 
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
        model_s3d = S3DM(var_y_shape).to(device)
        #
        if var_weight is not None:  model_s3d.load_state_dict(torch.load(var_weight))
        #
        optimizer = torch.optim.Adam(model_s3d.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_s3d, 
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
        model_s3d.load_state_dict(var_best_weight)
        #
        result_acc, result_dict, _ = test(model_s3d, 
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