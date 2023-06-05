# WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing





<img src="visualize/video_act_30_25.gif" height="280" title="hahahahahaha"/>  <img src="visualize/wifi_csi_act_30_25.gif" height="280"/>




<img src="visualize/video_act_49_41.gif" height="280"/>  <img src="visualize/wifi_csi_act_49_41.gif" height="280"/>




<img src="visualize/video_act_88_30.gif" height="280"/>  <img src="visualize/wifi_csi_act_88_30.gif" height="280"/>






```
mkdir dataset/wifi_csi/amp
python benchmark/wifi_csi/preprocess.py --dir_mat="dataset/wifi_csi/mat" --dir_amp="dataset/wifi_csi/amp"

python benchmark/wifi_csi/run.py --model="MLP" --task="activity" --repeat=2

python benchmark/video/preprocess.py --path_data_x="dataset/video" --path_data_y="dataset/annotation.csv" --model="Swin-T" --path_data_pre_x="dataset/cache/test"
```