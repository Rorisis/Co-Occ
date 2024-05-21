import os, sys
import cv2, imageio
import pickle, argparse
from mayavi import mlab
import mayavi
import copy
from typing import List, Optional, Tuple
import mmcv
from matplotlib import pyplot as plt
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
import numpy as np
import torch

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from tqdm import tqdm


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="gray",
        )

    # if bboxes is not None and len(bboxes) > 0:
    #     coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
    #     for index in range(coords.shape[0]):
    #         name = classes[labels[index]]
    #         plt.plot(
    #             coords[index, :, 0],
    #             coords[index, :, 1],
    #             linewidth=thickness,
    #             color=np.array(color or OBJECT_PALETTE[name]) / 255,
    #         )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="white",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2
    
    return coords_grid

#mlab.options.offscreen = True
occ_size = [200, 200, 16]
val_list = []
with open('./tools/nuscenes_val.txt', 'r') as file:
    for item in file:
        val_list.append(item[:-1])
file.close()

parser = argparse.ArgumentParser(description='')
parser.add_argument('pred_dir', default=None)
parser.add_argument('save_path', default=None)
args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval',
                dataroot='./data/nuscenes',
                verbose=True)
val_scenes = splits.val

grid = np.array(occ_size)
# for scene in tqdm(nusc.scene):
sample_files = os.listdir(args.pred_dir)
for index, sample_file in tqdm(enumerate(sample_files), total=len(sample_files)):
    if not sample_file[-3:] == 'pkl':
        continue

    sample_token = sample_file.split('.')[0]
    sample_file = os.path.join(args.pred_dir, sample_file)
    

    # continue
    print(sample_token)
    my_sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])

    # revise the visual path that required to visulize
    # visual_path = os.path.join('./data/nuscenes_occ/samples', lidar_path.split('/')[-1]+'.npy')
    visual_path = os.path.join('./visual_dir', lidar_path.split('/')[-1], 'pred.npy')
    
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    pred_voxels = sample_data['pred_voxels']
    cam2lidar = sample_data['cam2lidar']
    img_canvas = sample_data['img_canvas']

    
    
    voxel_size = [0.5, 0.5, 0.5]
    pc_range = [-50, -50,  -5, 50, 50, 3]
    vox_origin=np.array(pc_range[:3])

    # lidar_path = os.path.join('./data/nuscenes/samples/LIDAR_TOP', lidar_path.split('/')[-1])
    # points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)
    save_folder = os.path.join(args.save_path, '{}_assets'.format(sample_token))
    # visualize_lidar(
    #     os.path.join(save_folder, "lidar.png"),
    #     points,
    #     xlim=[pc_range[d] for d in [0, 3]],
    #     ylim=[pc_range[d] for d in [1, 4]],
    #     )

    # visual_path = os.path.join(args.pred_dir, lidar_path.split('/')[-1]+'.npy')
    # fov_voxels = np.load(visual_path)
    # fov_voxels[..., 3][fov_voxels[..., 3] == 0] = 255
    # voxels = np.zeros(grid)
    # voxels[fov_voxels[:, 0].astype(np.int), fov_voxels[:, 1].astype(np.int), fov_voxels[:, 2].astype(np.int)] = fov_voxels[:, 3]

    voxels = pred_voxels
    # voxels = fov_voxels
    w, h, z = voxels.shape
    grid = grid.astype(np.int32)
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    
    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    car_vox_range = np.array([
        [w//2 - 2 - 4, w//2 - 2 + 4],
        [h//2 - 2 - 4, h//2 - 2 + 4],
        [z//2 - 2 - 3, z//2 - 2 + 3]
    ], dtype=np.int32)
    
    ''' draw the colorful ego-vehicle '''
    car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    car_label = np.zeros([8, 8, 6], dtype=np.int32)
    car_label[:3, :, :2] = 17
    car_label[3:6, :, :2] = 18
    car_label[6:, :, :2] = 19
    car_label[:3, :, 2:4] = 18
    car_label[3:6, :, 2:4] = 19
    car_label[6:, :, 2:4] = 17
    car_label[:3, :, 4:] = 19
    car_label[3:6, :, 4:] = 17
    car_label[6:, :, 4:] = 18
    car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    grid_coords[car_indexes, 3] = car_label.flatten()

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    voxel_size = sum(voxel_size) / 3
    # # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]

    fov_voxels = np.load(visual_path)

    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]


    # #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene
    
    # plt_plot_fov = mlab.points3d(points[:,0], points[:,1], points[:,2],  color=(125/255, 125/255, 125/255), mode='point', colormap = 'gnuplot') 
    # plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    # plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    save_folder = args.save_path
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, '{}.png'.format(str(sample_token)))
    mlab.savefig(save_file)

    mlab.close()
# mlab.show()