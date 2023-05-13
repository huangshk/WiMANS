import time
import copy
import torch
import torch._dynamo
import numpy as np
#
from sklearn.metrics import classification_report, accuracy_score
from ptflops import get_model_complexity_info
#
from preset import preset

#
##
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
##
class ABLSTM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_input,
                 var_dim_output):
        #
        ##
        super(ABLSTM, self).__init__()


        #
        self.layer_bilstm = torch.nn.LSTM(input_size = var_dim_input,
                                          hidden_size = 512,
                                          batch_first = True,
                                          bidirectional = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(2*512, 2*512)
        self.layer_activation = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(2*512, var_dim_output)


        torch.nn.init.xavier_uniform_(self.layer_linear.weight)
        torch.nn.init.xavier_uniform_(self.layer_output.weight)
        #
        ##
        self.layer_softmax = torch.nn.Softmax(dim = -2)
        #
        ##
        self.layer_pooling = torch.nn.AvgPool1d(8, 8)
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_dropout = torch.nn.Dropout(0.6)  
    
        
    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        var_t = self.layer_norm(var_t)
        var_t = self.layer_pooling(var_t)
        var_t = torch.permute(var_t, (0, 2, 1))
        #
        var_h, _ = self.layer_bilstm(var_t)
        # print(var_h.shape)

        var_s = self.layer_linear(var_h)
        var_s = self.layer_activation(var_s)
        # print(var_s.shape)

        var_a = self.layer_softmax(var_s)

        var_t = var_h * var_a

        #
        var_t = torch.sum(var_t, dim = -2)
        # var_t = torch.mean(var_t, dim = -2)

        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_output(var_t)

        var_output = var_t
        #
        return var_output

    
#
##
def run_ablstm(data_train_x,
                 data_train_y,
                 data_test_x,
                 data_test_y,
                 var_repeat = 10):
    #
    ##
    #--------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------- Preprocess -------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------#
    #
    ##
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    var_shape_y = data_train_y[0].shape
    #
    data_train_y = data_train_y.reshape(data_train_y.shape[0], -1)
    data_test_y = data_test_y.reshape(data_test_y.shape[0], -1)
    #
    ##
    var_dim_x = data_train_x[0].shape[-1]
    var_dim_y = data_train_y[0].shape[-1]
    #
    ##
    data_train_set = torch.utils.data.TensorDataset(torch.from_numpy(data_train_x), 
                                                    torch.from_numpy(data_train_y))
    #
    data_train_loader = torch.utils.data.DataLoader(dataset = data_train_set, 
                                                    shuffle = True,
                                                    batch_size = preset["nn"]["batch_size"])
    #
    ##
    #--------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------- Train & Predict -----------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------#
    #
    ##
    var_macs, var_params = get_model_complexity_info(ABLSTM(var_dim_x, var_dim_y), 
                                                     data_train_x[0].shape, 
                                                     as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result = {}
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        torch.random.manual_seed(var_r + 39)
        #
        ##
        model_ablstm = ABLSTM(var_dim_x, var_dim_y).to(device)
        #                                
        model_ablstm = torch.compile(model_ablstm)
        #
        optimizer = torch.optim.Adam(model_ablstm.parameters(), 
                                     lr = preset["nn"]["lr"])
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1] * var_dim_y).to(device))
        #
        var_time_0 = time.time()
        #
        var_epochs = preset["nn"]["epoch"]
        #
        var_best_accuracy = 0
        var_best_weight = None
        #
        ##
        for var_epoch in range(var_epochs):
            #
            ##
            var_time_e0 = time.time()
            #
            model_ablstm.train()
            for data_batch in data_train_loader:
                #
                ##
                data_batch_x, data_batch_y = data_batch
                data_batch_x = data_batch_x.to(device)
                data_batch_y = data_batch_y.to(device)
                #
                predict_train_y = model_ablstm(data_batch_x)
                #
                var_loss_train = loss(predict_train_y, data_batch_y.float())
                #
                optimizer.zero_grad()
                #
                var_loss_train.backward()
                #
                optimizer.step()
            #
            ##
            model_ablstm.eval()
            #
            ##
            with torch.no_grad():
                #
                ##
                predict_test_y = model_ablstm(torch.from_numpy(data_test_x).to(device))
                #
                var_loss_test = loss(predict_test_y, torch.from_numpy(data_test_y).float().to(device))
                #
                predict_train_y = (torch.sigmoid(predict_train_y) > preset["nn"]["threshold"]).float()
                predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
                #
                ##
                data_batch_y = data_batch_y.detach().cpu().numpy()
                predict_train_y = predict_train_y.detach().cpu().numpy()
                predict_test_y = predict_test_y.detach().cpu().numpy()
                #
                ##
                data_batch_y_c = data_batch_y.reshape(-1, var_shape_y[-1]).astype(int)
                predict_train_y_c = predict_train_y.reshape(-1, var_shape_y[-1]).astype(int)
                var_accuracy_train = accuracy_score(data_batch_y_c, predict_train_y_c)
                #
                ##
                data_test_y_c = data_test_y.reshape(-1, var_shape_y[-1]).astype(int)
                predict_test_y_c = predict_test_y.reshape(-1, var_shape_y[-1]).astype(int)
                var_accuracy_test = accuracy_score(data_test_y_c, predict_test_y_c)
                #
                ##
                print(f"Epoch {var_epoch}/{var_epochs}",
                      "- %.6fs"%(time.time() - var_time_e0),
                      "- Loss %.6f"%var_loss_train.cpu(),
                      "- Accuracy %.6f"%var_accuracy_train,
                      "- Test Loss %.6f"%var_loss_test.cpu(),
                      "- Test Accuracy %.6f"%var_accuracy_test)
                #
                ##
                if var_accuracy_test > var_best_accuracy:
                    #
                    var_best_accuracy = var_accuracy_test
                    var_best_weight = copy.deepcopy(model_ablstm.state_dict())
        #
        ##
        var_time_1 = time.time()
        #
        model_ablstm.load_state_dict(var_best_weight)
        #
        predict_test_y = model_ablstm(torch.from_numpy(data_test_x).to(device))
        predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ##
        data_test_y_c = data_test_y.reshape(-1, var_shape_y[-1]).astype(int)
        predict_test_y_c = predict_test_y.reshape(-1, var_shape_y[-1]).astype(int)
        result_acc = accuracy_score(data_test_y_c, predict_test_y_c)
        #
        ##
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