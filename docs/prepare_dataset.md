## NuScenes
Please download **nuScenes full dataset v1.0**, **CAN bus expansion**, and **nuScenes-lidarseg** from the [official website](https://www.nuscenes.org/download). The dataset folder should be organized as follows:
```
Co-Occ
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── lidarseg (optional)
|   |   │   ├──v1.0-trainval/
|   |   │   ├──v1.0-mini/
|   |   │   ├──v1.0-test/
|   ├── nuscenes_infos_temporal_train.pkl
|   ├── nuscenes_infos_temporal_val.pkl
|   ├── nuscenes_infos_temporal_test.pkl
```

## NuScenes occupancy labels from SurroundOcc
Please download dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) from [here](https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md)

**Folder structure:**
```
SurroundOcc
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   ├── nuscenes_occ/
|   ├── nuscenes_infos_temporal_train.pkl
|   ├── nuscenes_infos_temporal_val.pkl
|   ├── nuscenes_infos_temporal_test.pkl
```


## (Optional) NuScenes occupancy labels from OpenOccupancy
Please download occupancy labels (resolution 512x512x40 with voxel size 0.2m) from [here](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md?plain=1)

**Folder structure:**
```
SurroundOcc
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   ├── nuScenes-Occupancy/
|   ├── nuscenes_infos_temporal_train.pkl
|   ├── nuscenes_infos_temporal_val.pkl
|   ├── nuscenes_infos_temporal_test.pkl

```

To generate the above data infos, directly download [infos](https://github.com/Rorisis/Co-Occ/releases/tag/data_infos) or prepare yourself by running:
```bash
python tools/create_data.py nuscenes --root-path ./data/ --out-dir ./data --extra-tag nuscenes --version v1.0 --canbus ./data
```

