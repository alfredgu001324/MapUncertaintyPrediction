## Merging Map Data with Trajectory Data

### Necessary Files
- `results.pickle` from mapping evaluation
- Trajdata pickle files: `traj_scene_frame_{full_train, full_val, mini_train, mini_val}.pkl`
- Ground truth files: `gt_{full_train, full_val, mini_val}.pickle`

You can download the above files from [here](https://drive.google.com/drive/folders/17kqpilI4dP6ZY7XFnWArPbMSlpDq8ErM?usp=drive_link)

### Data Merging Process
Run the script `adaptor.py` in `MapUncertaintyPrediction/adaptor/` to merge data mapping data and trajectory data. 

First create directories to hold the data
```
cd trj_data
mkdir maptr maptrv2 maptrv2_cent stream
cd ..
```

Then run

```
cd adaptor

# To merge MapTRv2 centerline, add in an additional argument --centerline
python adaptor.py \
  --version 'mini' \                                       # ['trainval', 'mini']
  --split 'mini_val' \                                     # ['train', 'train_val', 'val', 'mini_train', 'mini_val']
  --map_model 'MapTR' \                                    # ['MapTR', 'StreamMapNet']
  --dataroot '/home/data/nuscenes' \
  --index_file 'traj_scene_frame_mini_val.pkl' \
  --map_file 'results.pickle' \
  --gt_map_file 'gt_mini_val.pickle' \
  --save_path '../maptr' \                                 # trj_data directory

python adaptor_vis.py                                      # For visualization
```
