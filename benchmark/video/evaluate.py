"""
[file]          train.py
[description]   
"""
#
##
import time
import torch
#
from copy import deepcopy
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

#
##
def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: Dataset,
          data_test_set: Dataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device):
    #
    ##
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle = True, num_workers = 4)
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
            var_loss_train = loss(predict_train_y, data_batch_y.reshape(data_batch_y.shape[0], -1).float())
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            optimizer.step()
        #
        ##
        result_train_acc, _, _ = test(model, loss, data_train_set, var_threshold, var_batch_size, device)
        #
        result_test_acc, _, var_loss_test = test(model, loss, data_test_set, var_threshold, var_batch_size, device)
        #
        ##
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs"%(time.time() - var_time_e0),
              "- Loss %.6f"%var_loss_train.cpu(),
              "- Accuracy %.6f"%result_train_acc,
              "- Test Loss %.6f"%var_loss_test.cpu(),
              "- Test Accuracy %.6f"%result_test_acc)
        #
        ##
        if result_test_acc > var_best_accuracy:
            #
            var_best_accuracy = result_test_acc
            var_best_weight = deepcopy(model.state_dict())
    #
    ##
    return var_best_weight
    

#
##
def test(model: Module,
         loss: Module,
         data_set: Dataset,
         var_threshold: float,
         var_batch_size: int,
         device: device):
    #
    ##
    data_y = []
    predict_y = []
    #
    data_loader = DataLoader(data_set, var_batch_size, num_workers = 4)
    #
    ##
    model.eval()
    #
    with torch.no_grad():
        #
        ##
        for data_batch in data_loader:
            #
            data_batch_x, data_batch_y = data_batch
            #
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)
            #
            data_y.append(data_batch_y)
            predict_y.append(model(data_batch_x))       
    #
    ##
    data_y = torch.concat(data_y)
    predict_y = torch.concat(predict_y)
    #
    var_loss = loss(predict_y, data_y.reshape(data_y.shape[0], -1).float())
    #
    predict_y = (torch.sigmoid(predict_y) > var_threshold).float()
    #
    data_y = data_y.detach().cpu().numpy()
    predict_y = predict_y.detach().cpu().numpy()
    #
    predict_y = predict_y.reshape(-1, data_y.shape[-1]).astype(int)
    data_y = data_y.reshape(-1, data_y.shape[-1]).astype(int)
    #
    ## Accuracy
    result_acc = accuracy_score(data_y, predict_y)
    #
    ## Report
    result_dict = classification_report(data_y,
                                        predict_y, 
                                        digits = 6, 
                                        zero_division = 0, 
                                        output_dict = True)
    #
    return result_acc, result_dict, var_loss
        