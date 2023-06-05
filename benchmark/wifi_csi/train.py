"""
[file]          train.py
[description]   function to train WiFi-based models
"""
#
##
import time
import torch
import torch._dynamo
#
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score

#
##
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536

#
##
def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device):
    
    """
    [description]
    : generic training function for WiFi-based models
    [parameter]
    : model: Pytorch model to train
    : optimizer: optimizer to train model (e.g., Adam)
    : loss: loss function to train model (e.g., BCEWithLogitsLoss)
    : data_train_set: training set
    : data_test_set: test set
    : var_threshold: threshold to binarize sigmoid outputs
    : var_batch_size: batch size of each training step
    : var_epochs: number of epochs to train model
    : device: device (cuda or cpu) to train model
    [return]
    : var_best_weight: weights of trained model
    """
    #
    ##
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle = True, pin_memory = True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))
    #
    ##
    var_best_accuracy = 0
    var_best_weight = None
    #
    ##
    for var_epoch in range(var_epochs):
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_time_e0 = time.time()
        #
        model.train()
        #
        for data_batch in data_train_loader:
            #
            ##
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)
            #
            predict_train_y = model(data_batch_x)
            #
            var_loss_train = loss(predict_train_y, 
                                  data_batch_y.reshape(data_batch_y.shape[0], -1).float())
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            optimizer.step()
        #
        ##
        predict_train_y = (torch.sigmoid(predict_train_y) > var_threshold).float()
        #
        data_batch_y = data_batch_y.detach().cpu().numpy()
        predict_train_y = predict_train_y.detach().cpu().numpy()
        #
        predict_train_y = predict_train_y.reshape(-1, data_batch_y.shape[-1])
        data_batch_y = data_batch_y.reshape(-1, data_batch_y.shape[-1])
        var_accuracy_train = accuracy_score(data_batch_y.astype(int), 
                                            predict_train_y.astype(int))
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        model.eval()
        #
        with torch.no_grad():
            #
            ##
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            #
            predict_test_y = model(data_test_x)
            #
            var_loss_test = loss(predict_test_y, 
                                 data_test_y.reshape(data_test_y.shape[0], -1).float())
            #
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
            #
            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()
            #
            predict_test_y = predict_test_y.reshape(-1, data_test_y.shape[-1])
            data_test_y = data_test_y.reshape(-1, data_test_y.shape[-1])
            #
            var_accuracy_test = accuracy_score(data_test_y.astype(int), 
                                               predict_test_y.astype(int))
        #
        ## ---------------------------------------- Print -----------------------------------------
        #
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
            var_best_weight = deepcopy(model.state_dict())
    #
    ##
    return var_best_weight
