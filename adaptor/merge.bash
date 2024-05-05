python adaptor.py \
  --version mini \
  --split mini_val \
  --map_model MapTR \
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_mini_val.pkl \
  --map_file ../mapping_results.pickle \
  --gt_map_file ../adaptor_files/gt_mini_val.pickle \
  --save_path ../trj_data/maptrv2
