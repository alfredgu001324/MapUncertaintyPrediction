import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def normalize_lanes(x, y, lanes, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    n_lanes = []
    for lane in lanes:
        normalize_lane = np.dot(lane[:, :2] - np.array([x, y]), R)
        n_lanes.append(normalize_lane)
    
    return n_lanes

def normalize_point(x, y, point, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    normalize_point = np.dot(point - np.array([x, y]), R)
    
    return normalize_point

def normalize_traj(x, y, traj, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    normalize_traj = np.dot(traj[:, :2] - np.array([x, y]), R)
    
    return normalize_traj

def main(args):
    for filename in os.listdir(args.data_path):
        file_path = os.path.join(args.data_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        centerlines = data['gt_map']['centerlines']
        left_edges = data['gt_map']['left_edges']
        right_edges = data['gt_map']['right_edges']

        boundary_scores = data['predicted_map']['boundary_scores']
        ped_crossing_scores = data['predicted_map']['ped_crossing_scores']
        boundary_indices = np.where(boundary_scores > 0.4)[0]
        ped_crossing_indices = np.where(ped_crossing_scores > 0.4)[0]

        # boundary = data['maptr_gt_map']['boundary']
        # ped_crossing = data['maptr_gt_map']['ped_crossing']    
        boundary = np.array(data['predicted_map']['boundary'])[boundary_indices]
        ped_crossing = np.array(data['predicted_map']['ped_crossing'])[ped_crossing_indices]

        n_centerlines = normalize_lanes(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), centerlines, data['ego_heading'].item())
        n_left_edges = normalize_lanes(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), left_edges, data['ego_heading'].item())
        n_right_edges = normalize_lanes(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), right_edges, data['ego_heading'].item())

        n_boundary = normalize_lanes(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), boundary, data['ego_heading'].item())
        n_ped_crossing = normalize_lanes(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), ped_crossing, data['ego_heading'].item())

        fig, ax = plt.subplots(figsize=(6, 12))

        for lane in n_centerlines:
            plt.plot(lane[:, 1], lane[:, 0], c='blue')
        
        # for lane in n_left_edges:
        #     plt.plot(lane[:, 1], lane[:, 0], c='green')

        # for lane in n_right_edges:
        #     plt.plot(lane[:, 1], lane[:, 0], c='green')

        for lane in n_boundary:
            plt.plot(lane[:, 1], lane[:, 0], c='green')

        for lane in n_ped_crossing:
            plt.plot(lane[:, 1], lane[:, 0], c='black')
        
        plt.scatter(0, 0, s=100, c='red', marker="*")

        n_ego_hist = normalize_traj(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['ego_hist'], data['ego_heading'].item())
        n_ego_fut =  normalize_traj(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['ego_fut'], data['ego_heading'].item())
        plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='orange')
        plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

        for i in range(data['agent_hist'].shape[0]):
            color = {1: 'black', 2: 'orange', 3: 'black', 4: 'black'}
            n_point = normalize_point(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['agent_hist'][i, -1, :2],  data['ego_heading'].item())
            if -30 <= n_point[0] <= 30 and -15 <= n_point[1] <= 15:
                plt.scatter(n_point[1], n_point[0], s=100, c=color[data['agent_type'][i].item()], marker="*")
            else:
                continue

        # print(data['sample_token'])
        # print(data['scene_name'])

        ax.invert_xaxis()
        plt.show()
        # plt.savefig('/home/YOUR_USERNAME_HERE/project/plot_trajdata/' + data['sample_token'] + '.png', dpi=300)
        # plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Map with Trajectory')

    parser.add_argument('--data_path', type=str, required=True, help='save path')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)




