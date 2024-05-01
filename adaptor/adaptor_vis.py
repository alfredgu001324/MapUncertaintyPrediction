import pickle
import matplotlib.pyplot as plt
import numpy as np

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


# file_name = "/home/YOUR_USERNAME_HERE/project/pkl_files_adaptor_maptr_v2/vec_scene_full_val.pkl"
file_name = "/home/YOUR_USERNAME_HERE/project//vec_scene.pkl"
with open(file_name, 'rb') as file:
    data = pickle.load(file)


for key, value in data.items():
    centerlines = value['gt_map']['centerlines']
    left_edges = value['gt_map']['left_edges']
    right_edges = value['gt_map']['right_edges']

    # boundary = value['maptr_gt_map']['boundary']
    # ped_crossing = value['maptr_gt_map']['ped_crossing']    

    boundary_scores = value['predicted_map']['boundary_scores']
    ped_crossing_scores = value['predicted_map']['ped_crossing_scores']
    boundary_indices = np.where(boundary_scores > 0.4)[0]
    ped_crossing_indices = np.where(ped_crossing_scores > 0.4)[0]
    # breakpoint()
    boundary = np.array(value['predicted_map']['boundary'])[boundary_indices]
    ped_crossing = np.array(value['predicted_map']['ped_crossing'])[ped_crossing_indices]

    # breakpoint()

    n_centerlines = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), centerlines, value['ego_heading'].item())
    n_left_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), left_edges, value['ego_heading'].item())
    n_right_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), right_edges, value['ego_heading'].item())

    n_boundary = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), boundary, value['ego_heading'].item())
    n_ped_crossing = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), ped_crossing, value['ego_heading'].item())

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
    # breakpoint()

    n_ego_hist = normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_hist'], value['ego_heading'].item())
    n_ego_fut =  normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_fut'], value['ego_heading'].item())
    plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='orange')
    plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    for i in range(value['agent_hist'].shape[0]):
        color = {1: 'black', 2: 'orange', 3: 'black', 4: 'black'}
        n_point = normalize_point(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['agent_hist'][i, -1, :2],  value['ego_heading'].item())
        if -30 <= n_point[0] <= 30 and -15 <= n_point[1] <= 15:
            plt.scatter(n_point[1], n_point[0], s=100, c=color[value['agent_type'][i].item()], marker="*")
        else:
            continue

    # print(value['sample_token'])
    # print(value['scene_name'])

    ax.invert_xaxis()
    plt.show()
    # plt.savefig('/home/YOUR_USERNAME_HERE/project/plot_trajdata/' + value['sample_token'] + '.png', dpi=300)
    # plt.close(fig)
    # breakpoint()

# breakpoint()




