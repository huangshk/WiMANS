import time
import numpy as np
#
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#
##
def run_rf(data_train_x, 
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
    data_train_y = data_train_y.reshape(data_train_y.shape[0], -1)
    #
    ##
    var_standard_encoder = StandardScaler(with_std = None)
    #
    ## shape (samples, time_steps, channels)
    data_train_x = np.array([var_standard_encoder.fit_transform(data_sample_x) for data_sample_x in data_train_x])
    data_test_x = np.array([var_standard_encoder.fit_transform(data_sample_x) for data_sample_x in data_test_x])
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
    #--------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------- Train & Predict -----------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------#
    #
    ##
    result_precision = []
    result_time_train = []
    result_time_test = []
    result_dict = {}
    #
    ##
    for var_r in range(var_repeat):
        #
        ## model define
        model_rf = RandomForestClassifier(n_estimators = 10, 
                                          random_state = var_r + 39, 
                                          bootstrap = False) 
        #
        ## model training
        var_time_0 = time.time()
        model_rf.fit(data_train_ft_x, data_train_y)
        var_time_1 = time.time()
        #
        predict_test_y = model_rf.predict(data_test_ft_x)
        var_time_2 = time.time()
        #
        ## model predict
        predict_test_y = predict_test_y.reshape(data_test_y.shape)
        #
        result = classification_report(data_test_y.reshape(-1, data_test_y.shape[-1]), 
                                       predict_test_y.reshape(-1, predict_test_y.shape[-1]), 
                                       digits = 6, 
                                       zero_division = 0, 
                                       output_dict = True)
        #
        result_dict["repeat_" + str(var_r)] = result
        #
        result_precision.append(result["macro avg"]["precision"])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        #
    result_dict["mAP"] = {"avg": np.mean(result_precision), "std": np.std(result_precision)}
    result_dict["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result_dict["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    #
    return result_dict

