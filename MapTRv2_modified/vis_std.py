import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from nuscenes.utils.geometry_utils import view_points
from matplotlib.axes import Axes
from typing import Tuple, List
import matplotlib.lines as mlines
import cv2
import imageio
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import matplotlib.lines as mlines
import re
import random
import warnings
import argparse
import matplotlib.gridspec as gridspec
from PIL import Image

def average_of_tuples(tuple_list):
    if not tuple_list:  
        return ()

    sum_tuple = [0] * len(tuple_list[0])
    for tup in tuple_list:
        for i, value in enumerate(tup):
            sum_tuple[i] += value
    
    num_tuples = len(tuple_list)
    average_tuple = tuple(sum_val / num_tuples for sum_val in sum_tuple)
    
    return average_tuple

def normalize_lanes(x, y, lanes, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    n_lanes = []
    for lane in lanes:
        normalize_lane = np.dot(lane[:, :2] - np.array([x, y]), R)
        n_lanes.append(normalize_lane)
    
    return n_lanes

def normalize_traj(x, y, traj, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    normalize_traj = np.dot(traj[:, :2] - np.array([x, y]), R)
    
    return normalize_traj

def normalize_hivt(hivt_trj, x, y, theta):
    translation = np.array([x, y])
    
    translated = hivt_trj[:,:,:,:2] - translation

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    normalized_trajectories = np.einsum('klnj,ji->klni', translated, rotation_matrix)

    normalized_trajectories[:, :, :, 1] = -normalized_trajectories[:, :, :, 1]

    return normalized_trajectories

def normalize_agent(agent_trj, x, y, theta):
    translation = np.array([x, y])
    
    translated = agent_trj[:,:,:2] - translation

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    normalized_trajectories = np.einsum('ijk,kl->ijl', translated, rotation_matrix)

    normalized_trajectories[:, :, 1] = -normalized_trajectories[:, :, 1]

    return normalized_trajectories

def calc_metrics_min(prediction, gt):
    last_predicted = prediction[:, -1, :]  # Shape (6, 2)
    fde = np.linalg.norm(last_predicted - gt[-1], axis=1)  # Shape (6,)
    min_fde = np.min(fde)
    mode = np.argmin(fde)
    best = prediction[mode]
    min_ade = np.mean(np.linalg.norm(best - gt, axis=1), axis=0)
    mr = 1 if min_fde > 2.0 else 0

    return (min_ade, min_fde, mr)

def plot_points_with_laplace_variances(x, y, beta_x, beta_y, color, sample_idx, ax, heading, std):
    for i in range(len(x)):
        ax.plot(x[i], y[i], color=color, linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=color, s=2, alpha=0.8, zorder=-1)

        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        for j in range(len(x[i])):
            if std:
                width = np.sqrt(var_x[i][j])*2
                height = np.sqrt(var_y[i][j])*2
            else:
                width, height = 0, 0
            ellipse = Ellipse((x[i][j], y[i][j]), width=width, height=height,
                              fc=color, lw=0.5, alpha=0.3) 
            ax.add_patch(ellipse)

def render(box,
            axis: Axes,
            view: np.ndarray = np.eye(3),
            normalize: bool = False,
            colors: Tuple = ('b', 'r', 'k'),
            linewidth: float = 2,
            box_idx=None,
            alpha=0.5,
            y_offset: float = 1) -> None:
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color, alpha):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot([prev[0], corner[0]], [prev[1] + y_offset, corner[1] + y_offset], color=color, linewidth=linewidth, alpha=alpha)
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                    [corners.T[i][1]+y_offset, corners.T[i + 4][1]+y_offset],
                    color=colors[2], linewidth=linewidth, alpha=alpha)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0], alpha)
    draw_rect(corners.T[4:], colors[1], alpha)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0) + [0, y_offset]
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0) + [0, y_offset]
    axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth, alpha=alpha)
    if box_idx is not None and center_bottom[0] > -35 and center_bottom[1] > -35 \
        and center_bottom[0] < 35 and center_bottom[1] < 35:
        text = f'{box_idx}'
        axis.text(center_bottom[0], center_bottom[1], text, ha='left', fontsize=5)

def fig_to_image(fig):
    fig.set_dpi(300)
    
    fig.set_size_inches(fig.get_figwidth(), fig.get_figheight())
    fig.canvas.draw()  
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf

def create_gif_with_subplots(frames, output_path):
    """ Save a series of images as a GIF. """
    with imageio.get_writer(output_path, mode='I', fps=5) as writer:
        for frame in frames:
            writer.append_data(frame)

def create_mp4_with_subplots(frames, output_path):
    """ Save a series of images as an MP4 video. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with imageio.get_writer(output_path, fps=5, format='FFMPEG', codec='libx264', pixelformat='yuv420p') as writer:
            for frame in frames:
                writer.append_data(frame)

def place_colored_text(axi, text, x, y, fontdict):
    parts = re.split(r'(\(\s*[+-]?\d+%\))', text)
    current_x = x
    for i, part in enumerate(parts):
        color = 'black'  # Default color for text
        if re.match(r'\(\s*-\d+%\)', part):
            color = 'green'
        # Check for positive percentage
        elif re.match(r'\(\s*\+\d+%\)', part) and not re.match(r'\(\s*\+0+%\)', part):
            color = '#FF4500'
        # Check for zero percentage
        elif re.match(r'\(\s*0+%\)', part):
            color = 'grey'  # Explicitly setting color to black for 0%
        
        # Create text element
        if i == 0:
            if len(parts) > 1:
                text_artist = axi.text(current_x - 0.1, y, part, color=color, va='top', ha='center', transform=axi.transAxes, **fontdict)
            else:
                text_artist = axi.text(current_x, y, part, color=color, va='top', ha='center', transform=axi.transAxes, **fontdict)
        else:
            text_artist = axi.text(current_x + 0.33, y, part, color=color, va='top', ha='center', transform=axi.transAxes, **fontdict)

def main(args):
    tokens = []
    nusc = NuScenes(version=f'v1.0-{args.version}', dataroot=args.dataroot, verbose=True)
    splits = create_splits_scenes()

    sample_tokens_all = [s['token'] for s in nusc.sample]
    scene_frames_all = []
    scene_frames = []
    scene_descr = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[args.split]:
            if sample_token == scene_record['first_sample_token']:
                scene_frames = []
            scene_frames.append(sample_token)
            if sample_token == scene_record['last_sample_token']:
                scene_frames_all.append(scene_frames)
                scene_descr.append(scene_record['description'])

    with open("scene_descrip.txt", "w") as file:
        for index, item in enumerate(scene_descr):
            file.write(f"{index}: {item}\n")

    HiVT = False
    if args.trj_pred == 'HiVT':
        HiVT = True
    MapTR = False
    if args.map == 'MapTR':
        MapTR = True

    folder_path = args.trj_data

    data = {}

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.pkl'):
            scene_id = filename.replace('scene-', '').replace('.pkl', '')

            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'rb') as file:
                data_content = pickle.load(file)

            data[int(scene_id)] = data_content

    with open(args.base_results, 'rb') as handle:
        predict_data = pickle.load(handle)

    with open(args.unc_results, 'rb') as handle:
        predict_data_unc = pickle.load(handle)

    with open(args.boxes, 'rb') as handle:
        boxes_gt_all = pickle.load(handle)

    for key, value in predict_data.items():
        # HiVT
        if HiVT == True:
            x, y = data[int(key)]['ego_pos'][0, 0].item(), data[int(key)]['ego_pos'][0, 1].item()
            heading = data[int(key)]['ego_heading'].item()
            data[int(key)]['predict_fut'] = normalize_hivt(value, x, y, heading)
            data[int(key)]['predict_fut_std'] = normalize_hivt(predict_data_unc[key], x, y, heading)

        # DenseTNT
        else:
            data[int(key)]['predict_fut'] = np.flip(value.reshape(1, 6, 30, 2), -1)
            data[int(key)]['predict_fut_std'] = np.flip(predict_data_unc[key].reshape(1, 6, 30, 2), -1)


    track = {}
    for key, value in data.items():
        for i, tokens in enumerate(scene_frames_all):
            for j, token in enumerate(tokens):
                if value['sample_token'] == token:
                    track[(i, j)] = key

    frames = []
    for first_index, tokens in enumerate(tqdm(scene_frames_all)):
        metric_baseline_scene = []
        metric_unc_scene = []
        for second_index, token in enumerate(tokens):
            if (first_index, second_index) not in track:
                continue
            else:
                key = track[(first_index, second_index)]
                value = data[key]
                x, y = value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item()
                heading = value['ego_heading'].item()

                n_ego_fut =  normalize_traj(x, y, value['ego_fut'], heading)
                n_ego_fut[:,1] = -n_ego_fut[:,1]

                metric_baseline_scene.append(calc_metrics_min(value['predict_fut'][0], n_ego_fut))
                metric_unc_scene.append(calc_metrics_min(value['predict_fut_std'][0], n_ego_fut))

        average_baseline = average_of_tuples(metric_baseline_scene)
        average_unc = average_of_tuples(metric_unc_scene)

        improves = [(-(average_baseline[0] - average_unc[0])/(average_baseline[0]+0.00001)) * 100, 
            (-(average_baseline[1] - average_unc[1])/(average_baseline[1]+0.00001)) * 100, 
            (-(average_baseline[2] - average_unc[2])/(average_baseline[2]+0.00001)) * 100,
        ]
        for i, improve in enumerate(improves):
            if round(improve, 0) == 0:
                improves[i] = '0%'
            else:
                if improve > 0:
                    improves[i] = f'+{improve:.0f}%'
                elif improve < 0:
                    improves[i] = f'{improve:.0f}%'
        texts = [
            ("Baseline", f"minADE$_3$: {average_baseline[0]:.2f}", f"minFDE$_3$: {average_baseline[1]:.2f}", f"MR$_3$: {average_baseline[2]:.2f}"),
            ("Ours", f"minADE$_3$: {average_unc[0]:.2f} ({improves[0]})", f"minFDE$_3$: {average_unc[1]:.2f} ({improves[1]})", f"MR$_3$: {average_unc[2]:.2f} ({improves[2]})"),
            ("GT", ),
        ]

        for second_index, token in enumerate(tokens):
            if (first_index, second_index) not in track:
                continue
            else:
                key = track[(first_index, second_index)]
                value = data[key]

                x, y = value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item()
                heading = value['ego_heading'].item()
                fig, ax = plt.subplots(1, 3, figsize=(6, 6))
                x_min, x_max = -15, 15
                y_min, y_max = -30, 30  

                for ax_i in ax:
                    ax_i.set_xlim(x_min, x_max)
                    ax_i.set_ylim(y_min, y_max)

                ax[0].axis('off')
                ax[1].axis('off')
                ax[2].axis('off')

                divider_scores = value['predicted_map']['divider_scores']
                ped_crossing_scores = value['predicted_map']['ped_crossing_scores']
                boundary_scores = value['predicted_map']['boundary_scores']

                divider_indices = np.where(divider_scores > 0.4)[0]
                ped_crossing_indices = np.where(ped_crossing_scores > 0.4)[0]
                boundary_indices = np.where(boundary_scores > 0.4)[0]

                divider = np.array(value['predicted_map']['divider'])[divider_indices]
                ped_crossing = np.array(value['predicted_map']['ped_crossing'])[ped_crossing_indices]
                boundary = np.array(value['predicted_map']['boundary'])[boundary_indices]

                divider_betas = value['predicted_map']['divider_betas'][divider_indices]
                ped_crossing_betas = value['predicted_map']['ped_crossing_betas'][ped_crossing_indices]
                boundary_betas = value['predicted_map']['boundary_betas'][boundary_indices]

                if divider.size != 0:
                    n_divider = np.array(normalize_lanes(x, y, divider, heading))
                    n_divider = n_divider[:, :, [1, 0]]
                    n_divider[:,:,0] = -n_divider[:,:,0]
                    plot_points_with_laplace_variances(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax[0], heading, False)
                    plot_points_with_laplace_variances(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax[1], heading, True)
                if ped_crossing.size != 0:
                    n_ped_crossing = np.array(normalize_lanes(x, y, ped_crossing, heading))
                    n_ped_crossing = n_ped_crossing[:, :, [1, 0]]
                    n_ped_crossing[:,:,0] = -n_ped_crossing[:,:,0]
                    plot_points_with_laplace_variances(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax[0], heading, False)
                    plot_points_with_laplace_variances(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax[1], heading, True)
                if boundary.size != 0:
                    n_boundary = np.array(normalize_lanes(x, y, boundary, heading))
                    n_boundary = n_boundary[:, :, [1, 0]]
                    n_boundary[:,:,0] = -n_boundary[:,:,0]
                    plot_points_with_laplace_variances(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax[0], heading, False)
                    plot_points_with_laplace_variances(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax[1], heading, True)
                
                offset = 1

                n_ego_hist = normalize_traj(x, y, value['ego_hist'], heading)
                n_ego_fut =  normalize_traj(x, y, value['ego_fut'], heading)
                n_ego_hist[:,1] = -n_ego_hist[:,1]
                n_ego_fut[:,1] = -n_ego_fut[:,1]
                for i in range(3):
                    ax[i].plot(n_ego_hist[:, 1], n_ego_hist[:, 0] - offset, c='#489ACC')
                    ax[i].plot(n_ego_fut[:, 1], n_ego_fut[:, 0] - offset, c='red')

                origin = torch.tensor(value['ego_hist'][-1], dtype=torch.float)
                av_heading_vector = origin - torch.tensor(value['ego_hist'][-2], dtype=torch.float)
                theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
                rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                        [torch.sin(theta), torch.cos(theta)]])
                agent_indices_original = np.where(value['agent_type'] == 1)[0]
                keep_indices = []
                flag = False
                for index in agent_indices_original:
                    n_point = torch.matmul(torch.tensor([value['agent_hist'][index, -1, :2]]) - origin, rotate_mat)
                    if -30 < n_point[0][0].item() < 30 and -15 < n_point[0][1] < 15:
                        keep_indices.append(index)

                if HiVT:
                    if len(keep_indices) > 0:
                        n_agent_hist = normalize_agent(value['agent_hist'][keep_indices][:, :, :2], x, y, heading)
                        n_agent_fut = normalize_agent(value['agent_fut'][keep_indices], x, y, heading)
                        for i in range(len(n_agent_hist)):
                            for j in range(3):
                                ax[j].plot(n_agent_hist[i][:, 1], n_agent_hist[i][:, 0] - offset, c='#489ACC')
                                ax[j].plot(n_agent_fut[i][:, 1], n_agent_fut[i][:, 0] - offset, c='red')

                for i in range(len(value['predict_fut'])):
                    agent_trj = value['predict_fut'][i]
                    agent_trj_std = value['predict_fut_std'][i]

                    for j in range(6):
                        ax[0].plot(agent_trj[j][:, 1], agent_trj[j][:, 0] - offset, c='pink')
                        ax[1].plot(agent_trj_std[j][:, 1], agent_trj_std[j][:, 0] - offset, c='pink')

                # GT Map Plotting
                sample_token = value['sample_token']
                colors_plt = {'divider': 'orange', 'ped_crossing': 'b', 'boundary': 'g'}
                for k , v in value['maptr_gt_map'].items():
                    gt_map_element = np.array(normalize_lanes(x, y, np.array(v), heading))
                    if gt_map_element.shape == (0,):
                        continue
                    else:
                        gt_map_element = gt_map_element[:, :, [1, 0]]
                    gt_map_element[:,:,0] = -gt_map_element[:,:,0]
                    x_pts = gt_map_element[:, :, 0]
                    y_pts = gt_map_element[:, :, 1]

                    for j in range(len(x_pts)):
                        ax[2].plot(x_pts[j], y_pts[j], color=colors_plt[k], linewidth=1, alpha=0.8, zorder=-1)
                        ax[2].scatter(x_pts[j], y_pts[j], color=colors_plt[k], s=2, alpha=0.8, zorder=-1)

                boxes_gt = boxes_gt_all[sample_token]
                ignore_list = ['barrier', 'traffic_cone']
                conf_th = 0.4
                for i, box in enumerate(boxes_gt):  
                    box.score = 1
                    if box.name in ignore_list:
                        continue
                    # Show only predictions with a high score.
                    assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
                    if box.score < conf_th or abs(box.center[0]) > 15 or abs(box.center[1]) > 30:
                        continue
                    for j in range(3):
                        render(box, ax[j], view=np.eye(4), colors=('grey', 'grey', 'grey'), linewidth=1, box_idx=None, y_offset=0)
                
                for i in range(3):
                    ax[i].plot([-0.9, -0.9], [-2 - offset, 2 - offset], color='red', linewidth=1, alpha=0.8)
                    ax[i].plot([-0.9, 0.9], [2 - offset, 2 - offset], color='red', linewidth=1, alpha=0.8)
                    ax[i].plot([0.9, 0.9], [2 - offset, -2 - offset], color='red', linewidth=1, alpha=0.8)
                    ax[i].plot([0.9, -0.9], [-2 - offset, -2 - offset], color='red', linewidth=1, alpha=0.8)
                    ax[i].plot([0.0, 0.0], [0.0 - offset, 2 - offset], color='red', linewidth=1, alpha=0.8)

                # Font settings
                main_text_font = {'fontsize': 'large', 'fontweight': 'bold', 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
                small_text_font = {'fontsize': 'medium', 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}

                for i, axi in enumerate(ax):
                    axi.axis('off')  
                    # Main title
                    if i == len(ax) - 1:
                        axi.text(0.5, 1.11, texts[i][0], transform=axi.transAxes, ha='center', va='top', **main_text_font)
                        break
                    else:
                        axi.text(0.5, 1.21, texts[i][0], transform=axi.transAxes, ha='center', va='top', **main_text_font)
                    # Smaller texts
                    for j, subtext in enumerate(texts[i][1:]):
                        place_colored_text(axi, subtext, 0.5, 1.16 - j*0.05, small_text_font)

                colors = {
                    'darkgreen': '#00CC33',  
                    'darkyellow': '#F4BF00',  
                    'historyblue': '#5C99C7',  
                }

                legend_items = [
                    (colors['darkgreen'], 'Road Boundary'),
                    ('#FF0000', 'GT Future'),
                    (colors['darkyellow'], 'Lane Divider'),
                    ('#FFC0CB', 'Predicted Trajectories'),
                    ('#0000FF', 'Pedestrian Crossing'),  
                    (colors['historyblue'], 'Agent History'),
                ]

                legend_handles = [mlines.Line2D([], [], color=color, linewidth=4, label=label) for color, label in legend_items]

                fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3)

                plt.tight_layout()
                fig.subplots_adjust(top=0.8, bottom=0.11)  

                # plt.show()
                frame = fig_to_image(fig)
                frames.append(frame)
                plt.close(fig) 
        
        gif_dir = os.path.join(args.save_path, 'gif')
        mp4_dir = os.path.join(args.save_path, 'mp4')
        os.makedirs(gif_dir, exist_ok=True)
        os.makedirs(mp4_dir, exist_ok=True)

        gif_path = os.path.join(gif_dir, f'{first_index}.gif')
        mp4_path = os.path.join(mp4_dir, f'{first_index}.mp4')

        create_gif_with_subplots(frames, gif_path)
        create_mp4_with_subplots(frames, mp4_path)

        frames = []

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--version', type=str, default='trainval', choices=['trainval', 'mini'], help='version of nuscenes')
    parser.add_argument('--dataroot', type=str, required=True, help='directory of nuscenes raw data')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'mini_val'])

    parser.add_argument('--trj_pred', type=str, default='HiVT', choices=['HiVT', 'DenseTNT'], help='trj prediction model')
    parser.add_argument('--map', type=str, default='MapTR', choices=['MapTR', 'StreamMapNet'])

    parser.add_argument('--trj_data', type=str, required=True, help='processed data from adaptor')
    parser.add_argument('--base_results', type=str, required=True, help='trj prediction pkl of baseline approach')
    parser.add_argument('--unc_results', type=str, required=True, help='trj prediction pkl of unc approach')
    parser.add_argument('--boxes', type=str, required=True, help='gt boxes data path')

    parser.add_argument('--save_path', type=str, required=True, help='gif save path')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    main(args)