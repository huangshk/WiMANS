# Dataset Directory

Please extract the WiMANS dataset under this directory.

- Step 1: Download the WiMANS dataset from [[Kaggle]](https://www.kaggle.com/datasets/shuokanghuang/wimans)

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
    |   |   | - act_1_1.mat   # raw CSI sample labeled "act_1_1"
    |   |   | - act_1_2.mat   # raw CSI sample labeled "act_1_2"
    |   |   | - ...           # totally 11286 raw CSI samples (*.mat files)
    |   | - amp
    |   |   | - act_1_1.npy   # CSI amplitude labeled "act_1_1"
    |   |   | - act_1_2.npy   # CSI amplitude labeled "act_1_2"
    |   |   | - ...           # totally 11286 samples of CSI amplitude (*.npy files)
    | - video
    |   | - act_1_1.mp4       # video sample labeled "act_1_1"
    |   | - act_1_2.mp4       # video sample labeled "act_1_2"
    |   | - ...               # totally 11286 video samples (*.mp4 files)
    ```

Annotations are saved in the "annotation.csv" file, which can be read using Pandas.

```python
import pandas as pd
data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
# "var_path_data_y" is the path of "annotation.csv"
```

Raw CSI data are saved in "*.mat" files, which can be read using SciPy.

```python
import scipy.io as scio
data_mat = scio.loadmat(var_path_mat)
# "var_path_mat" is the path of "*.mat" file
```

The preprocessed data of CSI amplitude are saved in "*.npy" files, which can be read using NumPy.

```python
import numpy as np
data_csi = np.load(var_path)
# "var_path" is the path of "*.npy" file
```

Video data are saved in "*.mp4" files, which can be read using PyTorch.

```python
import torchvision
data_video_x, _, _ = torchvision.io.read_video(var_path, output_format = "TCHW")
# "var_path" is the path of "*.mp4" file
```



