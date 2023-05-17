import time
import torch
import numpy as np
from torchvision.models.video import resnet

from ptflops import get_model_complexity_info
from preset import preset

from train import train, test

#
##
def run_resnet(data_train_set,
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
    var_macs, var_params = get_model_complexity_info(resnet.r3d_18(num_classes = var_y_shape[-1]), 
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
        model_resnet = resnet.r3d_18(num_classes = var_y_shape[-1]).to(device)
        #
        if var_weight is not None:  model_resnet.load_state_dict(torch.load(var_weight))
        #
        optimizer = torch.optim.Adam(model_resnet.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_resnet, 
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
        model_resnet.load_state_dict(var_best_weight)
        #
        result_acc, result_dict, _ = test(model_resnet, 
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