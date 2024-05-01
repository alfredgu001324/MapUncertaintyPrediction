from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import pickle

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trajdata import MapAPI
from trajdata import SceneBatch, UnifiedDataset
from trajdata.caching.scene_cache import SceneCache

import argparse

def normalize_lanes(x, y, lanes, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    n_lanes = []
    for lane in lanes:
        normalize_lane = np.dot(lane[:, :2] - np.array([x, y]), R)
        n_lanes.append(normalize_lane)
    
    return n_lanes

def denormalize_lanes(x, y, lanes, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    n_lanes = []
    for lane in lanes:
        normalize_lane = np.dot(lane[:, :2], R.T) + np.array([x, y])
        n_lanes.append(normalize_lane)
    
    return n_lanes

def lanes_within_bound(lanes, width, height):
    qualified_lanes = []
    for lane in lanes:
        x_conditions = (lane[:, 0] > -width/2) & (lane[:, 0] < width/2)
        y_conditions = (lane[:, 1] > -height/2) & (lane[:, 1] < height/2)
        to_append = lane[x_conditions & y_conditions]
        if len(to_append) > 1:
            qualified_lanes.append(to_append)

    return qualified_lanes
    
def get_vec_map(map_api, map_name, scene_cache, mean_pt, heading):
    vec_map = map_api.get_map(f"{map_name}", scene_cache=scene_cache,
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True)

    width = 60
    height = 30
    radius = np.sqrt((width/2)**2 + (height/2)**2)

    lanes = vec_map.get_lanes_within(mean_pt, radius)
    centerlines = [lane.center.points for lane in lanes]
    left_boundary = [lane.left_edge.interpolate(num_pts=100).points for lane in lanes if lane.left_edge is not None]
    right_boundary = [lane.right_edge.interpolate(num_pts=100).points for lane in lanes if lane.right_edge is not None]

    n_centerlines = normalize_lanes(mean_pt[0], mean_pt[1], centerlines, heading)
    bounded_centerlines = lanes_within_bound(n_centerlines, width, height)
    world_bounded_centerlines = denormalize_lanes(mean_pt[0], mean_pt[1], bounded_centerlines, heading)

    n_left_edge = normalize_lanes(mean_pt[0], mean_pt[1], left_boundary, heading)
    bounded_left_edge = lanes_within_bound(n_left_edge, width, height)
    world_bounded_left_edge = denormalize_lanes(mean_pt[0], mean_pt[1], bounded_left_edge, heading)

    n_right_edge = normalize_lanes(mean_pt[0], mean_pt[1], right_boundary, heading)
    bounded_right_edge = lanes_within_bound(n_right_edge, width, height)
    world_bounded_right_edge = denormalize_lanes(mean_pt[0], mean_pt[1], bounded_right_edge, heading)

    return {
        "centerlines": world_bounded_centerlines,
        "left_edges": world_bounded_left_edge,
        "right_edges": world_bounded_right_edge
        }

def scale_values(values, old_min, old_max, new_min, new_max):
    return new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)

def scale_stream(data):
    pts = data['vectors']
    betas = data['betas']
    scores = data['scores']
    indices = np.where(scores > 0)[0]

    x = pts[indices, :, 0]  # x values
    y = pts[indices, :, 1]  # y values

    old_x_min, old_x_max = 0, 1
    new_x_min, new_x_max = -30, 30

    old_y_min, old_y_max = 1, 0    #0, 1
    new_y_min, new_y_max = -15, 15

    x_scaled = np.empty_like(x)
    y_scaled = np.empty_like(y)

    x_scaled = scale_values(x, old_x_min, old_x_max, new_x_min, new_x_max)
    y_scaled = scale_values(y, old_y_min, old_y_max, new_y_min, new_y_max)

    beta_x = betas[indices, :, 0] * 60
    beta_y = betas[indices, :, 1] * 30
    new_betas = np.stack((beta_y, beta_x), axis=-1)
    new_pts = np.stack((y_scaled, x_scaled), axis=-1) 

    return new_pts, new_betas

def main(args):

    dataset = UnifiedDataset(
        desired_data=[f"nusc_{args.version}-{args.split}"],
        centric="scene",
        desired_dt=0.1,
        incl_robot_future=True,
        incl_raster_map=True, # True
        # incl_vector_map=True,
        standardize_data=False,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=4,
        verbose=True,
        data_dirs={  
            f"nusc_{args.version}": args.datroot,
        },
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    scene_cache: Optional[SceneCache] = None

    batch: SceneBatch
    data = {}
    scene_frame_count = {}


    file_name = args.index_file
    with open(file_name, 'rb') as file:
        scene_data = pickle.load(file)

    scenario_count = 0
    for batch in tqdm(dataloader):
        ego_hist_len = batch.agent_hist_len[:, 0]
        ego_fut_len = batch.agent_fut_len[:, 0]

        scene_frame_count[batch.scene_ids[0]] = len(scene_data[batch.scene_ids[0]])
        values = [(value-1) for value in scene_frame_count.values()]
        cumulative_sums = [cumsum*5 + i for i, cumsum in enumerate(np.cumsum(values))]    

        if ego_hist_len < 20 or ego_fut_len < 30:
            continue
        
        else:
            idx = batch.data_idx
            dt = batch.dt

            cum_index = np.searchsorted(cumulative_sums, idx)
            scene_name = list(scene_frame_count.keys())[cum_index.item()]
            if cum_index == 0:
                frame_idx = idx/5 
            else:
                frame_idx = (idx - cumulative_sums[cum_index.item()-1] - 1) / 5
            
            if frame_idx.item().is_integer():
                pass
            else:
                continue

            sample_token = scene_data[scene_name][frame_idx.item()]

            agent_hist = batch.agent_hist.squeeze(0)[1:, -20:, [0, 1, 3, 4, 7]]   # #agents, time_len, (x, y, vx, vy, heading)
            agent_fut = batch.agent_fut.squeeze(0)[1:, :30, :2]    # #agents, time_len, (x,y)
            agent_type = batch.agent_type[:, 1:].ravel()

            valid_hist = np.where(~np.any(np.isnan(agent_hist.numpy()), axis=(1, 2)))[0]
            valid_fut = np.where(~np.any(np.isnan(agent_fut.numpy()), axis=(1, 2)))[0]
            valid_agent_idx = np.intersect1d(valid_hist.ravel(), valid_fut.ravel())

            agent_hist = agent_hist[valid_agent_idx]
            agent_fut = agent_fut[valid_agent_idx]
            agent_type = agent_type[valid_agent_idx]

            ego_type = batch.agent_type[:, 0]
            ego_pos = batch.centered_agent_state.position # x, y  equal to last position of ego_hist
            ego_heading = batch.centered_agent_state.heading   # 0 on +ve x axis, bottom plane is negative, upper plane positive

            ego_hist = batch.agent_hist.squeeze(0)[0, -20:, :2]
            ego_fut = batch.agent_fut.squeeze(0)[0, :30, :2]

            map_name = batch.map_names[0]   
            vec_map = get_vec_map(map_api, map_name, scene_cache, torch.cat((ego_pos, torch.tensor([[0]])), dim=1).squeeze(0).numpy(), ego_heading.item())
            
            data[scenario_count] = {
                'dt': dt.numpy(),                             # 0.1 s
                'agent_type': agent_type.numpy(),             # VEHICLE = 1 PEDESTRIAN = 2 BICYCLE = 3 MOTORCYCLE = 4
                'agent_hist': agent_hist.numpy(),             # num_agents, 20, 2
                'agent_fut': agent_fut.numpy(),               # num_agents, 30, 2
                'ego_type': ego_type.numpy(),                 # VEHICLE = 1
                'ego_pos': ego_pos.numpy(),                   # (x, y)
                'ego_heading': ego_heading.numpy(),           # 0 -> -2pi: bottom plane; 0 -> pi: upper plane
                'ego_hist': ego_hist.numpy(),                 # 30, 2
                'ego_fut': ego_fut.numpy(),                   # 20, 2
                'map_name': map_name,                         # eg. boston-seaport
                'gt_map': vec_map,                            # dict.keys() = ['centerlines', 'left_edges', 'right_edges']
                'scene_name': scene_name,                     # eg. scene-0103
                'sample_token': sample_token,                 # temporal sample token
            }
            scenario_count += 1
    
    print("Collected {} scenarios. ".format(scenario_count))
    
    sample_to_idx = {}
    for i in range(len(data)):
        sample_to_idx[data[i]['sample_token']] = i
    
    with open(args.map_file, 'rb') as handle:
        map_data = pickle.load(handle)

    for i in tqdm(range(len(map_data)), desc="Merging Map Estimation"):
        predicted_map = {}

        if args.map_model == "MapTR":
            sample_token = map_data[i]['pts_bbox']['sample_idx']
        elif args.map_model == "StreamMapNet":
            sample_token = map_data[i]['token']

        try:
            idx = sample_to_idx[sample_token]
        except:
            continue

        if args.map_model == "MapTR":
            divider_index = np.where(map_data[i]['pts_bbox']['labels_3d'] == 0)
            ped_crossing_index = np.where(map_data[i]['pts_bbox']['labels_3d'] == 1)
            boundary_index = np.where(map_data[i]['pts_bbox']['labels_3d'] == 2)

            predict_divider = map_data[i]['pts_bbox']['pts_3d'][divider_index].numpy()
            predict_ped_crossing = map_data[i]['pts_bbox']['pts_3d'][ped_crossing_index].numpy()
            predict_boundary = map_data[i]['pts_bbox']['pts_3d'][boundary_index].numpy()

            predict_divider_betas = map_data[i]['pts_bbox']['betas_3d'][divider_index].numpy()
            predict_ped_crossing_betas = map_data[i]['pts_bbox']['betas_3d'][ped_crossing_index].numpy()
            predict_boundary_betas = map_data[i]['pts_bbox']['betas_3d'][boundary_index].numpy()

            predict_divider_scores = map_data[i]['pts_bbox']['scores_3d'][divider_index].numpy()
            predict_ped_crossing_scores = map_data[i]['pts_bbox']['scores_3d'][ped_crossing_index].numpy()
            predict_boundary_scores = map_data[i]['pts_bbox']['scores_3d'][boundary_index].numpy()

        elif args.map_model == "StreamMapNet":
            map_data[i]['vectors'], map_data[i]['betas'] = scale_stream(map_data[i])
            divider_index = np.where(map_data[i]['labels'] == 1)
            ped_crossing_index = np.where(map_data[i]['labels'] == 0)
            boundary_index = np.where(map_data[i]['labels'] == 2)

            predict_divider = map_data[i]['vectors'][divider_index]
            predict_ped_crossing = map_data[i]['vectors'][ped_crossing_index]
            predict_boundary = map_data[i]['vectors'][boundary_index]

            predict_divider_betas = map_data[i]['betas'][divider_index]
            predict_ped_crossing_betas = map_data[i]['betas'][ped_crossing_index]
            predict_boundary_betas = map_data[i]['betas'][boundary_index]

            predict_divider_scores = map_data[i]['scores'][divider_index]
            predict_ped_crossing_scores = map_data[i]['scores'][ped_crossing_index]
            predict_boundary_scores = map_data[i]['scores'][boundary_index]

        x, y = data[idx]['ego_pos'][0, 0].item(), data[idx]['ego_pos'][0, 1].item()
        heading = data[idx]['ego_heading'].item()

        predict_divider[:,:,0] = -predict_divider[:,:,0]
        predict_divider = predict_divider[:, :, [1, 0]]
        dn_predict_divider = denormalize_lanes(x, y, predict_divider, heading)

        predict_ped_crossing[:,:,0] = -predict_ped_crossing[:,:,0]
        predict_ped_crossing = predict_ped_crossing[:, :, [1, 0]]
        dn_predict_ped_crossing = denormalize_lanes(x, y, predict_ped_crossing, heading)

        predict_boundary[:,:,0] = -predict_boundary[:,:,0]
        predict_boundary = predict_boundary[:, :, [1, 0]]
        dn_predict_boundary = denormalize_lanes(x, y, predict_boundary, heading)

        if args.centerline:
            centerline_index = np.where(map_data[i]['pts_bbox']['labels_3d'] == 3)
            predict_centerlines = map_data[i]['pts_bbox']['pts_3d'][centerline_index].numpy()
            predict_centerlines[:,:,0] = -predict_centerlines[:,:,0]
            predict_centerlines = predict_centerlines[:, :, [1, 0]]
            dn_predict_centerlines = denormalize_lanes(x, y, predict_centerlines, heading)
            predict_centerlines_betas = map_data[i]['pts_bbox']['betas_3d'][centerline_index].numpy()
            predict_centerlines_scores = map_data[i]['pts_bbox']['scores_3d'][centerline_index].numpy()

            predicted_map = {
                'divider': dn_predict_divider,
                'ped_crossing': dn_predict_ped_crossing,
                'boundary': dn_predict_boundary,
                'centerline': dn_predict_centerlines,
                'divider_betas': predict_divider_betas,
                'ped_crossing_betas': predict_ped_crossing_betas,
                'boundary_betas': predict_boundary_betas,
                'centerline_betas': predict_centerlines_betas,
                'divider_scores': predict_divider_scores,
                'ped_crossing_scores': predict_ped_crossing_scores,
                'boundary_scores': predict_boundary_scores,
                'centerline_scores': predict_centerlines_scores,
            }
        else:
            predicted_map = {
                'divider': dn_predict_divider,
                'ped_crossing': dn_predict_ped_crossing,
                'boundary': dn_predict_boundary,
                'divider_betas': predict_divider_betas,
                'ped_crossing_betas': predict_ped_crossing_betas,
                'boundary_betas': predict_boundary_betas, 
                'divider_scores': predict_divider_scores,
                'ped_crossing_scores': predict_ped_crossing_scores,
                'boundary_scores': predict_boundary_scores,
            }

        data[idx]['predicted_map'] = predicted_map
    
    with open(args.gt_map_file, 'rb') as handle:
        gt_data = pickle.load(handle)

    for key, value in tqdm(gt_data.items(), desc="Merging MapTR GT"):
        sample_token = key
        try:
            idx = sample_to_idx[sample_token]
        except:
            continue

        divider_index = np.where(value['gt_labels'] == 0)
        ped_crossing_index = np.where(value['gt_labels'] == 1)
        boundary_index = np.where(value['gt_labels'] == 2)

        gt_divider = value['gt_lines'][divider_index]
        gt_ped_crossing = value['gt_lines'][ped_crossing_index]
        gt_boundary = value['gt_lines'][boundary_index]

        x, y = data[idx]['ego_pos'][0, 0].item(), data[idx]['ego_pos'][0, 1].item()
        heading = data[idx]['ego_heading'].item()

        gt_divider[:,:,0] = -gt_divider[:,:,0]
        gt_divider = gt_divider[:, :, [1, 0]]
        dn_gt_divider = denormalize_lanes(x, y, gt_divider, heading)

        gt_ped_crossing[:,:,0] = -gt_ped_crossing[:,:,0]
        gt_ped_crossing = gt_ped_crossing[:, :, [1, 0]]
        dn_gt_ped_crossing = denormalize_lanes(x, y, gt_ped_crossing, heading)

        gt_boundary[:,:,0] = -gt_boundary[:,:,0]
        gt_boundary = gt_boundary[:, :, [1, 0]]
        dn_gt_boundary = denormalize_lanes(x, y, gt_boundary, heading)

        # This is the gt map from MapTR series, the above gt map is from trajdata
        maptr_gt_map = {
                'divider': dn_gt_divider,
                'ped_crossing': dn_gt_ped_crossing,
                'boundary': dn_gt_boundary,
            }

        data[idx]['maptr_gt_map'] = maptr_gt_map
    

    base_dir = args.save_path
    scenes_dir = os.path.join(base_dir, args.split)

    if not os.path.exists(scenes_dir):
        os.makedirs(scenes_dir)

    for i in range(len(data)):
        save_path = os.path.join(scenes_dir, 'scene-' + str(i) + '.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(data[i], file)

    # Store in one single pickle file
    # save_path = '/home/YOUR_USERNAME_HERE/code/vec_scene.pkl'
    # with open(save_path, 'wb') as file:
    #     pickle.dump(data, file)


def parse_args():
    parser = argparse.ArgumentParser(description='Merge Map with Trajectory')

    parser.add_argument('--version', type=str, default='trainval', choices=['trainval', 'mini'], help='version of nuscenes')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'train_val', 'val', 'mini_train', 'mini_val'])

    parser.add_argument('--map_model', type=str, required=True choices=['MapTR', 'StreamMapNet'])
    parser.add_argument('--centerline', action='store_true', help='centerline usage')

    parser.add_argument('--dataroot', type=str, required=True, help='directory of nuscenes raw data')

    parser.add_argument('--index_file', type=str, required=True, help='scene index file')
    parser.add_argument('--map_file', type=str, required=True, help='estimated maps data')
    parser.add_argument('--gt_map_file', type=str, required=True, help='gt map data')

    parser.add_argument('--save_path', type=str, required=True, help='save path')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)
