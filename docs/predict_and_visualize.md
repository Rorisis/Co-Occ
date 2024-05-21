# Prediction and Visualization with Co-Occ

For visualization, we follow two steps: (1) We inference the semantic occupancy with camera/LiDAR/multi-modal inputs and save the results. (2) We use mayavi to visualize the predictions. To install mayavi, please follow the [official instructions](http://docs.enthought.com/mayavi/mayavi/).

The above steps can be simplified by removing the saving and loading process if your server is able to perform both tasks, but we can only provide the two-step practices here.

## 1. nuScenes

### 1.1 Key-frame

To inference with 8 GPUs on the validation set (with R50, for example), run:
```bash
bash tools/dist_test.sh projects/configs/coocc_nusc/coocc_multi_r50_256x704.py $YOUR_CKPT 8 --pred-save $YOUR_SAVE_PATH
```

Then visualize with the following command:
```bash
python projects/mmdet3d_plugin/visualize/visualize_nusc.py $YOUR_SAVE_PATH $YOUR_VISUALIZE_SAVE_PATH
```
or
```bash
python projects/mmdet3d_plugin/visualize/visualize_nusc_gt.py $YOUR_SAVE_PATH $YOUR_VISUALIZE_SAVE_PATH
```

### 1.2 Video demo

First, we need to reorganize the data infos to include camera sweeps. Run:
```bash
python projects/mmdet3d_plugin/tools/prepare_video_infos.py --src-path data/nuscenes_infos_temporal_val.pkl --dst-path data/nuscenes_infos_temporal_val_visualize.pkl --data-path data/nuscenes
```

Then, inference on these contiguous camera sweeps by changing the data_info:
```bash
bash tools/dist_test.sh projects/configs/coocc_nusc/coocc_multi_r50_256x704.py $YOUR_CKPT 8 --pred-save $YOUR_SAVE_PATH --cfg-options data.test.ann_file=data/nuscenes_infos_temporal_val_visualize.pkl
```

Finally visualize with the following command:
```bash
python projects/mmdet3d_plugin/visualize/visualize_nusc_video.py $YOUR_SAVE_PATH $YOUR_VISUALIZE_SAVE_PATH
```
You can specify the scene-name by `--scene-name scene-0519`, for example, otherwise the script will browse all scene folders under the save_path.

## 2. Notes

The above visualization with mayavi follows the offline paradigm. To visualize each sample online for more detailed interaction, you can set `mlab.options.offscreen = False` and add `mlab.show()` before `mlab.savefig()`.
