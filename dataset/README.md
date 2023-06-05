# Dataset Directory

Please download the WiMANS dataset and extract it under this directory.

- Step 1: Download the WiMANS dataset from [[Kaggle]](https://www.kaggle.com/datasets/c4ecbbf66f200ced9ad8b7d2e3c0371c6e615ef2ee203174f09bcefb7a12d523)

- Step 2: Extract the entire WiMANS dataset under this directory

	```sh
	upzip dataset.zip
	```

- Step 3: Make sure the extracted WiMANS dataset follows such a file structure

    ```sh
    dataset
    | - README.md
    | - annotation.csv        # labels, annotations (e.g., user identities, locations, activities)
    | - wifi_csi
    |   | - mat
    |   |   |   act_1_1.mat   # raw CSI sample labeled "act_1_1"
    |   |   |   act_1_2.mat   # raw CSI sample labeled "act_1_2"
    |   |   |   ...           # totally 11286 raw CSI samples (*.mat files)
    |   | - amp
    |   |   |   act_1_1.npy   # CSI amplitude labeled "act_1_1"
    |   |   |   act_1_2.npy   # CSI amplitude labeled "act_1_2"
    |   |   |   ...           # totally 11286 samples of CSI amplitude (*.npy files)
    | - video
    |   | - act_1_1.mp4       # video sample labeled "act_1_1"
    |   | - act_1_2.mp4       # video sample labeled "act_1_2"
    |   | - ...               # totally 11286 video samples (*.mp4 files)
    ```



