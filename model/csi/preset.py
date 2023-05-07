#
##
preset = {
    #
    ##
    "path": {
        "data_x": "data/csi/amp",
        "data_y": "data/annotation.csv",
        "save": "result_activity_mlp_classroom_5.json"
    },
    #
    ##
    "data": {
        "num_users": ["0", "1", "2", "3", "4", "5"],        # "0", "1", "2", "3", "4", "5"
        "wifi_band": ["5"],                                 # "2.4", "5"
        "environment": ["classroom"],                       # "classroom", "meeting_room", "empty_room"
        "length": 3000,
    },
    #
    ##
    "task": "activity",                                     # identity, activity, location
    #
    ##
    "nn": {
        "lr": 1e-4, # 1e-3
        "epoch": 200,
        "batch_size": 128,
        "threshold": 0.5,
    },
    #
    ##
    "encoding": {
        "activity": {
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
        "location": {
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
}
