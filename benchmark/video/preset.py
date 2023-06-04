"""
[file]          preset.py
[description]   default settings of video-based models
"""
#
##
preset = {
    #
    ## define the model
    "model": "Swin-T",                                  # identity, activity, location
    #
    ## define the task
    "task": "identity",                                 # identity, activity, location
    #
    ## number of repeated experiments
    "repeat": 10,
    #
    ## path of data
    "path": {
        "data_x": "dataset/video",                      # directory of video files 
        "data_pre_x": "dataset/cache/resnet",           # directory of preprocessed video files 
        "data_y": "dataset/annotation.csv",             # path of the annotation file
        "save_result": "result.json",                   # path to save the results
        "save_model": None,                             # path to save/load the model
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0", "1", "2", "3", "4", "5"],    # select the number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "environment": ["classroom"],                   # select the environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 1e-4,                                     # learning rate
        "epoch": 20,                                    # number of epochs
        "batch_size": 8,                                # batch size
        "threshold": 0.5,                               # threshold to binarize the sigmoid outputs
        # "repeat": 1,
        "frame_stride": 1,                              # stride to downsample video frames
    },
    #
    ## the encoding of activities and locations
    "encoding": {
        "activity": {                                   # the encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "location": {                                   # the encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
}