"""
[file]          train.py
[description]   
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
from torch.utils.data import DataLoader
from numpy import ndarray
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
          data_train_loader: DataLoader,
          data_test_x: ndarray,
          data_test_y: ndarray,
          var_shape_y: tuple,
          var_threshold: float,
          var_epochs: int,
          device: device):
    #
    ##
    var_best_accuracy = 0
    var_best_weight = None
    #
    ##
    for var_epoch in range(var_epochs):
        #
        ##
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
            var_loss_train = loss(predict_train_y, data_batch_y.float())
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            optimizer.step()
        #
        ##
        model.eval()
        #
        with torch.no_grad():
            #
            ##
            predict_test_y = model(torch.from_numpy(data_test_x).to(device))
            #
            var_loss_test = loss(predict_test_y, torch.from_numpy(data_test_y).float().to(device))
            #
            predict_train_y = (torch.sigmoid(predict_train_y) > var_threshold).float()
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
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
            var_best_weight = deepcopy(model.state_dict())
    #
    ##
    return var_best_weight
        