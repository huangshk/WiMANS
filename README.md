# WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing

This repository provides all the [data](https://www.kaggle.com/datasets/c4ecbbf66f200ced9ad8b7d2e3c0371c6e615ef2ee203174f09bcefb7a12d523), source code, and documents of WiMANS. To the best of our knowledge, WiMANS is the first WiFi-based **multi-user** activity sensing dataset based on WiFi Channel State Information (CSI). WiMANS contains 11286 CSI samples of dual WiFi bands (2.4 / 5 GHz) and synchronized videos for reference and unexplored tasks (e.g., multi-user pose estimation). Each 3-second sample includes 0 to 5 users performing identical/different activities simultaneously, annotated with (anonymized) user identities, locations, and activities.

<table align = "center">
  <tr align = "center"><td rowspan="2"> <b>Sample <br/> "act_30_25" </b></td><td>WiFi CSI (5GHz)</td> <td>Synchronized Video</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_30_25.gif" height="188"/></td><td><img src="visualize/video_act_30_25.gif" height="188"></td></tr>
</table>

<table align = "center">
  <tr align = "center"><td rowspan="2"> <b>Sample <br/> "act_49_41"</b></td><td>WiFi CSI (2.4GHz)</td> <td>Synchronized Video</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_49_41.gif" height="188"/></td><td><img src="visualize/video_act_49_41.gif" height="188"/></td></tr>
</table>

<table align = "center">
  <tr align = "center"><td rowspan="2"> <b>Sample <br/> "act_88_30" </b></td><td>WiFi CSI (2.4GHz)</td> <td>Synchronized Video</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_88_30.gif" height="188"/></td><td><img src="visualize/video_act_88_30.gif" height="188"/></td></tr>
</table>



## Environment

- Ubuntu 20.04
- Python 3.9.12
- SciPy 1.7.3
- NumPy 1.21.5
- Pandas 1.4.2
- PyTorch 2.0.1





## Dataset Directory

Please download the WiMANS dataset and extract it under the "dataset" directory.

- Step 1: Download the WiMANS dataset from [[Kaggle]](https://www.kaggle.com/datasets/c4ecbbf66f200ced9ad8b7d2e3c0371c6e615ef2ee203174f09bcefb7a12d523)

- Step 2: Extract the entire WiMANS dataset under this directory

  ```sh
  upzip dataset.zip
  ```

- Step 3: Make sure the extracted WiMANS dataset follows such a file structure

    ```
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



## Benchmark Experiments

### WiFi-based Models



```sh
python benchmark/wifi_csi/preprocess.py --dir_mat="dataset/wifi_csi/mat" --dir_amp="dataset/wifi_csi/amp"
```


```sh
python benchmark/wifi_csi/run.py --model="MLP" --task="activity" --repeat=10
```


### Video-based Models
```sh
python benchmark/video/preprocess.py --path_data_x="dataset/video" --path_data_y="dataset/annotation.csv" --model="ResNet" --path_data_pre_x="dataset/cache/test"
```

```sh
python benchmark/video/run.py --model="ResNet" --task="activity" --repeat=10
```

