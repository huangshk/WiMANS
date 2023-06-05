"""
[file]          strf.py
[description]   implement and evaluate WiFi-based model ST-RF
"""
#
##
import time
import numpy as np
#
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#
##
def run_strf(data_train_x, 
             data_train_y,
             data_test_x,
             data_test_y,
             var_repeat = 10):
    """
    [description]
    : run WiFi-based model ST-RF
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ##
    var_standard_encoder = StandardScaler(with_std = None)
    #
    ## shape (samples, time_steps, channels)
    data_train_x = np.array([var_standard_encoder.fit_transform(data_sample_x) 
                             for data_sample_x in data_train_x])
    data_test_x = np.array([var_standard_encoder.fit_transform(data_sample_x) 
                            for data_sample_x in data_test_x])
    #
    data_train_time_x = np.swapaxes(data_train_x[:, :, :], -1, -2)
    data_test_time_x = np.swapaxes(data_test_x[:, :, :], -1, -2)
    #
    _, _, data_train_ft_x = signal.spectrogram(data_train_time_x, noverlap = 8, nperseg = 16, nfft = 16)
    _, _, data_test_ft_x = signal.spectrogram(data_test_time_x, noverlap = 8, nperseg = 16, nfft = 16)
    #
    data_train_ft_x = np.average(data_train_ft_x, axis = -1)
    data_train_ft_x = data_train_ft_x.reshape(data_train_ft_x.shape[0], -1)
    #
    data_test_ft_x = np.average(data_test_ft_x, axis = -1)
    data_test_ft_x = data_test_ft_x.reshape(data_test_ft_x.shape[0], -1)
    #
    ##
    ## ========================================= Train & Evaluate =========================================
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
        ## define model
        model_rf = RandomForestClassifier(n_estimators = 10, 
                                          random_state = var_r + 39, 
                                          bootstrap = False) 
        #
        ## model training
        var_time_0 = time.time()
        #
        model_rf.fit(data_train_ft_x, data_train_y.reshape(data_train_y.shape[0], -1))
        #
        var_time_1 = time.time()
        #
        ## model predict
        #
        predict_test_y = model_rf.predict(data_test_ft_x)
        #
        var_time_2 = time.time()
        #
        ##
        data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
        predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
        #
        result_acc = accuracy_score(data_test_y_c, predict_test_y_c)
        #
        result_dict = classification_report(data_test_y_c, predict_test_y_c, 
                                            digits = 6, zero_division = 0, output_dict = True)
        #
        result["repeat_" + str(var_r)] = result_dict
        #
        result_accuracy.append(result_acc)
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
    #
    ##
    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    #
    return result

