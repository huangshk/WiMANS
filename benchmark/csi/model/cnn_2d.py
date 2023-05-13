import time
import torch
import numpy as np
#
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from train import train
from preset import preset

#
##
class CNN_2D(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_input,
                 var_dim_output):
        #
        ##
        super(CNN_2D, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.BatchNorm2d(var_dim_input)
        self.layer_norm_1 = torch.nn.BatchNorm2d(32)
        self.layer_norm_2 = torch.nn.BatchNorm2d(64)
        self.layer_norm_3 = torch.nn.BatchNorm2d(128)
        #
        self.layer_cnn_2d_0 = torch.nn.Conv2d(in_channels = var_dim_input, 
                                              out_channels = 32,
                                              kernel_size = (27, 27),
                                              stride = (7, 7))
        #
        self.layer_cnn_2d_1 = torch.nn.Conv2d(in_channels = 32, 
                                              out_channels = 64,
                                              kernel_size = (15, 15),
                                              stride = (3, 3))
        #
        self.layer_cnn_2d_2 = torch.nn.Conv2d(in_channels = 64, 
                                              out_channels = 128,
                                              kernel_size = (7, 7),
                                              stride = (1, 1))
        #
        self.layer_linear = torch.nn.Linear(128, var_dim_output)
        #
        self.layer_relu = torch.nn.LeakyReLU()
        #
        self.layer_dropout = torch.nn.Dropout(0.2)
        #
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_linear.weight)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.unsqueeze(var_t, dim = 1)
        #
        var_t = self.layer_norm_0(var_t)
        var_t = self.layer_cnn_2d_0(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_1(var_t)
        var_t = self.layer_cnn_2d_1(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_2(var_t)
        var_t = self.layer_cnn_2d_2(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_3(var_t)
        var_t = torch.mean(var_t, dim = (-2, -1))
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output












#
##
def run_cnn_2d(data_train_x,
               data_train_y,
               data_test_x,
               data_test_y,
               var_repeat = 10):
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    var_shape_y = data_train_y[0].shape
    #
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    data_train_y = data_train_y.reshape(data_train_y.shape[0], -1)
    data_test_y = data_test_y.reshape(data_test_y.shape[0], -1)
    #
    var_dim_x, var_dim_y = 1, data_train_y[0].shape[-1]
    #
    ##
    data_train_set = torch.utils.data.TensorDataset(torch.from_numpy(data_train_x), 
                                                    torch.from_numpy(data_train_y))
    #
    data_train_loader = torch.utils.data.DataLoader(dataset = data_train_set, 
                                                    batch_size = preset["nn"]["batch_size"],
                                                    shuffle = True, pin_memory = True)
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(CNN_2D(var_dim_x, var_dim_y), 
                                                     data_train_x[0].shape, 
                                                     as_strings = False)
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
        model_cnn_2d = CNN_2D(var_dim_x, var_dim_y).to(device)
        #
        optimizer = torch.optim.Adam(model_cnn_2d.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 1e-4)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([6] * var_dim_y).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_cnn_2d, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_loader = data_train_loader, 
                                data_test_x = data_test_x, 
                                data_test_y = data_test_y, 
                                var_shape_y = var_shape_y, 
                                var_threshold = preset["nn"]["threshold"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device)
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_cnn_2d.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_cnn_2d(torch.from_numpy(data_test_x).to(device))
        predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ## Accuracy
        data_test_y_c = data_test_y.reshape(-1, var_shape_y[-1]).astype(int)
        predict_test_y_c = predict_test_y.reshape(-1, var_shape_y[-1]).astype(int)
        result_acc = accuracy_score(data_test_y_c, predict_test_y_c)
        #
        ## Report
        result_dict = classification_report(data_test_y_c, 
                                            predict_test_y_c, 
                                            digits = 6, 
                                            zero_division = 0, 
                                            output_dict = True)
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
    return result