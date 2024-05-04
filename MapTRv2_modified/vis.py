import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from nuscenes.nuscenes import NuScenes

with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptr.pkl', 'rb') as handle:
    data1 = pickle.load(handle)

with open('/home/YOUR_USERNAME_HERE/project/distance_data_stream.pkl', 'rb') as handle:
    data2 = pickle.load(handle)

with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptrv2.pkl', 'rb') as handle:
    data3 = pickle.load(handle)

with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptrv2_cent.pkl', 'rb') as handle:
    data4 = pickle.load(handle)

data = {'MapTR': data1, "StreamMapNet": data2, "MapTRv2": data3, "MapTRv2_cent": data4}
save_path = '/home/YOUR_USERNAME_HERE/project/distance_data.pkl'
# breakpoint()
with open(save_path, 'wb') as file:
    pickle.dump(data, file)
breakpoint()

def plot_points_with_ellipsoids(x, y, sigma_x, sigma_y, rho, labels):
    colors_plt = ['orange', 'b', 'g']
    fig, ax = plt.subplots(figsize=(6,12))

    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors_plt[labels[i]], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=colors_plt[labels[i]])
        # Draw an ellipse around each point
        for j in range(len(x[i])):
            # Calculate angle of rotation for the ellipse
            angle = np.arctan(rho[j]) * 180 / np.pi
            ellipse = Ellipse((x[i][j], y[i][j]), width=sigma_x[j]*2, height=sigma_y[j]*2,
                            angle=angle, edgecolor='red', fc='None', lw=2)
            ax.add_patch(ellipse)

    plt.show()

def calculate_std(x, y, beta_x, beta_y, labels, scores, sample_idx, nusc):
    colors_plt = ['orange', 'b', 'g', 'black']
    var_x = 2 * beta_x ** 2
    var_y = 2 * beta_y ** 2
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    # Calculate the L2 norm (Euclidean norm) for each element
    # uncertainties = np.sqrt(std_x**2 + std_y**2).numpy().flatten()

    uncertainties = np.sqrt(std_x**2 + std_y**2).numpy()
    max_uncertainties = np.max(uncertainties, axis=1)
    indices = np.argmax(uncertainties, axis=1)
    uncertainties = max_uncertainties

    # pos = np.stack((x, y), axis=2)
    # distances = np.sqrt(pos[:,:,0]**2 + pos[:,:,1]**2).flatten()
    distances = np.sqrt(x[np.arange(len(x)), indices].numpy()**2 + y[np.arange(len(y)), indices].numpy()**2).flatten()
    # distances = y.flatten()

    # Calculate the average uncertainty for each distance
    # breakpoint()
    average_uncertainty = np.mean(uncertainties)

    # new_labels = np.repeat(labels, 20).numpy()
    new_labels = labels.numpy()
    # breakpoint()

    # Create a line plot
    # plt.figure(figsize=(8, 6))
    # for i in range(len(distances)):
    #     plt.scatter(distances[i], uncertainties[i], color=colors_plt[new_labels[i]])

    # # Add a horizontal line for the average uncertainty
    # plt.axhline(average_uncertainty, color='red', linestyle='--', label='Average Uncertainty')

    # # Customize the plot
    # plt.xlabel('Distance')
    # plt.ylabel('Uncertainty')
    # plt.title('Uncertainty vs. Distance')
    # plt.legend()
    # # plt.show()
    # breakpoint()

    # Sample token
    sample_token = sample_idx

    # Fetch the sample record
    sample_record = nusc.get('sample', sample_token)

    # Get the scene token from the sample record
    scene_token = sample_record['scene_token']

    # Fetch the scene record
    scene_record = nusc.get('scene', scene_token)

    if 'night' in scene_record['description'].lower() or 'difficult lighting' in scene_record['description'].lower():
        night_tag = np.ones_like(new_labels)
    else:
        night_tag = np.zeros_like(new_labels)

    if 'rain' in scene_record['description'].lower():
        rain_tag = np.ones_like(new_labels)
    else:
        rain_tag = np.zeros_like(new_labels)

    if 'turn' in scene_record['description'].lower():
        turn_tag = np.ones_like(new_labels)
    else:
        turn_tag = np.zeros_like(new_labels)

    if 'intersection' in scene_record['description'].lower():
        intersection_tag = np.ones_like(new_labels)
    else:
        intersection_tag = np.zeros_like(new_labels)
    # Get the scene ID
    # scene_id = scene_record['name']
    # breakpoint()
    # print(scene_record['description'])
    tags = [night_tag, rain_tag, turn_tag, intersection_tag]
    return distances, uncertainties, new_labels, tags
    
    # breakpoint()

def get_speed(positions):
    # Time interval
    dt = 0.1

    # Calculate differences in position
    differences = np.diff(positions, axis=0)

    # Calculate velocity (change in position over time)
    velocity = differences / dt

    # If you need the magnitude of velocity (speed)
    speed = np.linalg.norm(velocity, axis=1)
    # breakpoint()
    return speed

def plot_points_with_laplace_variances(x, y, beta_x, beta_y, labels, scores, sample_idx):
    colors_plt = ['orange', 'b', 'g', 'black']
    # fig, ax = plt.subplots(figsize=(6, 12))
    fig, ax = plt.subplots(figsize=(2, 4))
    plt.axis('off')
    car_img = Image.open('/home/YOUR_USERNAME_HERE/project/final_plots/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors_plt[labels[i]], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=colors_plt[labels[i]], s=2, alpha=0.8, zorder=-1)
        # Calculate the variance from the beta values
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        # Draw an axis-aligned ellipse around each point
        for j in range(len(x[i])):
            # Using variance to set the width and height of the ellipse
            ellipse = Ellipse((x[i][j], y[i][j]), width=np.sqrt(var_x[i][j])*2, height=np.sqrt(var_y[i][j])*2,
                              fc=colors_plt[labels[i]], lw=0.5, alpha=0.5) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)
    plt.show()
    # plt.savefig('/home/YOUR_USERNAME_HERE/project/final_plots/maptrv2_cent/' + sample_idx + '.png', bbox_inches='tight', format='png',dpi=1200)
    plt.close(fig)


# with open('/home/YOUR_USERNAME_HERE/project/MapTR_v2_modified/results_centerline_24_full_val.pickle', 'rb') as handle:
with open('/home/YOUR_USERNAME_HERE/project/pkl_files_adaptor_maptr/results_full_val.pickle', 'rb') as handle:
    data = pickle.load(handle)

with open('/home/YOUR_USERNAME_HERE/project/pkl_files_adaptor_maptr/vec_scene_full_val.pkl', 'rb') as handle:
    motion_data = pickle.load(handle)

all_distances = np.array([])
all_uncertainties = np.array([])
all_labels = np.array([])
all_night_tags = np.array([])
all_rain_tags = np.array([])
all_turn_tags = np.array([])
all_intersection_tags = np.array([])
all_speeds = np.array([])
# Initialize the nuScenes API
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/data/nuscenes', verbose=True)

for i in tqdm(range(0, len(data))):
    pts = data[i]['pts_bbox']['pts_3d']
    betas = data[i]['pts_bbox']['betas_3d']
    scores = data[i]['pts_bbox']['scores_3d']
    labels = data[i]['pts_bbox']['labels_3d']
    sample_idx = data[i]['pts_bbox']['sample_idx']
    indices = np.where(scores > 0.4)[0]
    x = pts[indices, :, 0]  # x values
    y = pts[indices, :, 1]  # y values
    beta_x = betas[indices, :, 0]
    beta_y = betas[indices, :, 1]

    # plot_points_with_laplace_variances(x, y, beta_x, beta_y, labels[indices], scores[indices], sample_idx)
    # distances, uncertainties, labels = calculate_std(x, y, beta_x, beta_y, labels[indices], scores[indices], sample_idx, nusc)
    # all_distances = np.concatenate((all_distances, distances))
    # all_uncertainties = np.concatenate((all_uncertainties, uncertainties))
    # all_labels = np.concatenate((all_labels, labels))
    for key, value in motion_data.items():
        if value['sample_token'] == sample_idx:
            distances, uncertainties, labels, tags = calculate_std(y, x, beta_y, beta_x, labels[indices], scores[indices], sample_idx, nusc)
            speed = get_speed(value['ego_hist'])
            speeds = speed[-1] * np.ones_like(distances)
            
            night_tags = tags[0]
            rain_tags = tags[1]
            turn_tags = tags[2]
            intersection_tags = tags[3]
            all_distances = np.concatenate((all_distances, distances))
            all_uncertainties = np.concatenate((all_uncertainties, uncertainties))
            all_labels = np.concatenate((all_labels, labels))
            all_night_tags = np.concatenate((all_night_tags, night_tags))
            all_rain_tags = np.concatenate((all_rain_tags, rain_tags))
            all_turn_tags = np.concatenate((all_turn_tags, turn_tags))
            all_intersection_tags = np.concatenate((all_intersection_tags, intersection_tags))
            all_speeds = np.concatenate((all_speeds, speeds))
            break

# average_uncertainty = np.mean(all_uncertainties)
# colors_plt = ['orange', 'b', 'g', 'black']
# plt.figure(figsize=(8, 6))
# for i in range(len(all_distances[:5000])):
#     # breakpoint()
#     if int(all_labels[i].item()) == 3:
#         plt.scatter(all_distances[i], all_uncertainties[i], color=colors_plt[int(all_labels[i].item())])

# # Add a horizontal line for the average uncertainty
# plt.axhline(average_uncertainty, color='red', linestyle='--', label='Average Uncertainty')

# # Customize the plot
# plt.xlabel('Distance')
# plt.ylabel('Uncertainty')
# plt.title('Uncertainty vs. Distance')
# plt.legend()
# plt.show()

# Filter the data where int(all_labels) == 3
# indices = np.where(all_labels.astype(int) == 0)[0]
# filtered_distances = all_distances[indices]
# filtered_uncertainties = all_uncertainties[indices]
# filtered_labels = all_labels[indices]

all_tags = {'night_tag': all_night_tags, 'rain_tag': all_rain_tags, 'turn_tag': all_turn_tags, 'intersection_tag': all_intersection_tags}
filtered_distances = all_distances
filtered_uncertainties = all_uncertainties
filtered_labels = all_labels

# Fit a linear regression line
coefficients = np.polyfit(filtered_distances, filtered_uncertainties, 1)
slope, intercept = coefficients

# Create a line based on the regression
regression_line = slope * filtered_distances + intercept

# Plot the data points and the regression line
# plt.scatter(filtered_distances, filtered_uncertainties, alpha=0.1, color='b', label='Data', s=10)
# sns.stripplot(x=filtered_distances, y=filtered_uncertainties, hue=filtered_labels, dodge=True, jitter=True)
# sns.pointplot(x=filtered_distances, y=filtered_uncertainties, hue=filtered_labels, dodge=False, join=True)

# Define a mapping from existing labels to new categories
label_mapping = {
    0: 'Divider',
    1: 'Pedestrian Crossing',
    2: 'Boundary',
    3: 'Centerline'
}
new_labels = np.array([label_mapping[label] for label in filtered_labels])

# Define the bin edges for the x-axis
x_bins = np.arange(0, 35, 5)  # Bins every 2 meters from 0 to 20 meters
# custom_palette = ["#008000", "#000000", "#FFA500", "#0000FF"]
custom_palette = ["#008000", "#FFA500", "#0000FF"]

# Create a pointplot with Seaborn and bin the x-axis values
plt.figure(figsize=(12, 6))
ax = plt.gca()
sns.set_palette(custom_palette)
sns.pointplot(x=pd.cut(filtered_distances, bins=x_bins), y=filtered_uncertainties, hue=new_labels, dodge=True, ax=ax)

ax.legend(loc='upper left')
# plt.plot(filtered_distances, regression_line, color='r', label='Linear Regression')
plt.xticks(fontsize=23)  
plt.yticks(fontsize=23)
plt.xlabel('Distances (m)', fontsize=25)
plt.ylabel('Uncertainties (m)', fontsize=25)
plt.legend(fontsize=20)
plt.xticks(rotation=45)
# plt.show()
# plt.savefig('/home/YOUR_USERNAME_HERE/project/final_plots/distance/' + 'MapTR' + '.png', bbox_inches='tight', format='png',dpi=1200)
# plt.savefig('/home/YOUR_USERNAME_HERE/project/final_plots/distance/' + 'MapTR' + '.pdf', bbox_inches='tight', format='pdf',dpi=1200)

data = {'distance': filtered_distances, 'uncertainties': filtered_uncertainties, 'labels': new_labels, 'extra_tag': all_tags, 'speed': all_speeds}
save_path = '/home/YOUR_USERNAME_HERE/project/distance_data_maptr.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(data, file)

# with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptr.pkl', 'rb') as handle:
#     data1 = pickle.load(handle)

# with open('/home/YOUR_USERNAME_HERE/project/distance_data_stream.pkl', 'rb') as handle:
#     data2 = pickle.load(handle)

# with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptrv2.pkl', 'rb') as handle:
#     data3 = pickle.load(handle)

# with open('/home/YOUR_USERNAME_HERE/project/distance_data_maptrv2_cent.pkl', 'rb') as handle:
#     data4 = pickle.load(handle)

# data = {'MapTR': data1, "StreamMapNet": data2, "MapTRv2": data3, "MapTRv2_cent": data4}
# save_path = '/home/YOUR_USERNAME_HERE/project/distance_data.pkl'
# # breakpoint()
# with open(save_path, 'wb') as file:
#     pickle.dump(data, file)