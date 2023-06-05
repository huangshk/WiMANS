# WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing





<table align = "center">
  <tr align = "center"><td colspan="2"><b>Sample "act_30_25"</b></td>
  <tr align = "center"><td>WiFi CSI (5GHz) Data</td> <td>Video Data</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_30_25.gif" height="200"/></td><td><img src="visualize/video_act_30_25.gif" height="200"/></td></tr>
</table>

<table align = "center">
  <tr align = "center"><td colspan="2"><b>Sample "act_49_41"<\b></td>
  <tr align = "center"><td>WiFi CSI (2.4GHz) Data</td> <td>Video Data</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_49_41.gif" height="200"/></td><td><img src="visualize/video_act_49_41.gif" height="200"/></td></tr>
</table>

<table align = "center">
  <tr align = "center"><td colspan="2"><b>Sample "act_88_30"<\b></td>
  <tr align = "center"><td>WiFi CSI (2.4GHz) Data</td> <td>Video Data</td></tr>
  <tr align = "center"><td><img src="visualize/wifi_csi_act_88_30.gif" height="200"/></td><td><img src="visualize/video_act_88_30.gif" height="200"/></td></tr>
</table>


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
    
    
## WiFi-based Human Sensing
    
    

```
mkdir dataset/wifi_csi/amp
python benchmark/wifi_csi/preprocess.py --dir_mat="dataset/wifi_csi/mat" --dir_amp="dataset/wifi_csi/amp"

python benchmark/wifi_csi/run.py --model="MLP" --task="activity" --repeat=2
```
    
    
## Video-based Human Sensing
```
python benchmark/video/preprocess.py --path_data_x="dataset/video" --path_data_y="dataset/annotation.csv" --model="Swin-T" --path_data_pre_x="dataset/cache/test"
```
