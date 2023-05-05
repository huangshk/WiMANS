import time
import copy
import torch
import numpy as np
#
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, precision_score, accuracy_score
#
##
from preset import preset

#
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
##
class MLP(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_input,
                 var_dim_output):
        #
        ##
        super(MLP, self).__init__()
        #
        ##
        self.layer_0 = torch.nn.Linear(var_dim_input, 256)
        self.layer_1 = torch.nn.Linear(256, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, var_dim_output)
        #
        ##
        torch.nn.init.xavier_uniform_(self.layer_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        #
        ##
        # self.layer_relu = torch.nn.Tanh()
        self.layer_relu = torch.nn.ReLU()
        # self.layer_relu = torch.nn.Sigmoid()
        #
        ##
        self.layer_dropout = torch.nn.Dropout(0.1)
        #
        # self.layer_norm = torch.nn.LayerNorm(var_dim_input)
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_norm(var_t)
        #
        var_t = self.layer_0(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_1(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_2(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_3(var_t)
        #
        var_output = var_t
        #
        return var_output

#
##
def run_mlp(data_train_x, 
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
    data_train_x = data_train_x.reshape(data_train_x.shape[0], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], -1)
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
        model_mlp = MLP(var_dim_x, var_dim_y).to(device)
        #
        optimizer = torch.optim.Adam(model_mlp.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 1e-3)
        #
        loss = torch.nn.BCEWithLogitsLoss()
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
            model_mlp.train()
            for data_batch in data_train_loader:
                #
                ##
                data_batch_x, data_batch_y = data_batch
                data_batch_x = data_batch_x.to(device)
                data_batch_y = data_batch_y.to(device)
                #
                predict_train_y = model_mlp(data_batch_x)
                #
                var_loss_train = loss(predict_train_y, data_batch_y.float())
                #
                optimizer.zero_grad()
                #
                var_loss_train.backward()
                #
                # torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), 1.0)
                #
                optimizer.step()
            #
            ##
            model_mlp.eval()
            #
            ##
            with torch.no_grad():
                #
                ##
                predict_test_y = model_mlp(torch.from_numpy(data_test_x).to(device))
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
                      "- Loss %.6f"%var_loss_train.cpu(),
                      "- Accuracy %.6f"%var_accuracy_train,
                      "- Test Loss %.6f"%var_loss_test.cpu(),
                      "- Test Accuracy %.6f"%var_accuracy_test)
                #
                ##
                if var_accuracy_test > var_best_accuracy:
                    #
                    var_best_accuracy = var_accuracy_test
                    var_best_weight = copy.deepcopy(model_mlp.state_dict())
        #
        ##
        var_time_1 = time.time()
        #
        model_mlp.load_state_dict(var_best_weight)
        #
        predict_test_y = model_mlp(torch.from_numpy(data_test_x).to(device))
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
        print(result)
    #
    ##
    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    #
    return result