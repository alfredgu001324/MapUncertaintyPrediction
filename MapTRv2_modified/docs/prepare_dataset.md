

## NuScenes
Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_map_infos_temporal_{train,val}.pkl`, which contain local vectorized map annotations.

**Folder structure**
```
MapTR
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

## Argoverse2
Download the Argoverse2 Sensor Dataset [here](https://www.argoverse.org/av2.html#download-link).

**Folder structure**
```
MapTR
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   ├── argoverse2/
│   │   ├── sensor/
|   |   |   |—— train/
|   |   |   |—— val/
|   |   |   |—— test/
```

**Prepare Argoverse2 data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/maptrv2/custom_av2_map_converter.py --data-root ./data/argoverse2/sensor/
```

Using the above code will generate `av2_map_infos_{train,val}.pkl`, which contain local vectorized map annotations.
