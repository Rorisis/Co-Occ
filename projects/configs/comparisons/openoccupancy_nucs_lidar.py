_base_ = [
    '../_base_/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = None
occ_path = "./data/nuscenes_occ"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
occ_size = [200, 200, 16]
lss_downsample = [2, 2, 2]  # [128 128 10]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

cascade_ratio = 2
sample_from_voxel = True
sample_from_img = False

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes'
nusc_class_metas = 'projects/configs/_base_/nuscenes.yaml'
file_client_args = dict(backend='disk')

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    # 'input_size': (256, 704),
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}
numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='OccNet',
    loss_norm=True,
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.125, 0.125, 0.125],  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=numC_Trans,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[800, 800, 64],  # hardcode, xy size follow centerpoint
        ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=10000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(0, 0),
            scale_lim=(1, 1),
            flip_dx_ratio=0,
            flip_dy_ratio=0)

train_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_OccFormer', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', dataset='nusc'),
    dict(type='LoadOccupancy', is_train=True, to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask, bda_aug_conf=bda_aug_conf, cls_metas=nusc_class_metas),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
     dict(type='LoadMultiViewImageFromFiles_OccFormer', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', dataset='nusc'),
    dict(type='LoadOccupancy', is_train=True, to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask, bda_aug_conf=bda_aug_conf, cls_metas=nusc_class_metas),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['gt_occ', 'points'],
            meta_keys=['pc_range', 'occ_size', 'sample_idx', 'timestamp',
                       'scene_token', 'img_filenames', 'scene_name']),
]


test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='data/nuscenes_infos_temporal_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        occ_size=occ_size,
        pc_range=point_cloud_range),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
#     train=train_config,
#     val=test_config,
#     test=test_config,
#     shuffler_sampler=dict(type='DistributedGroupSampler'),
#     nonshuffler_sampler=dict(type='DistributedSampler'),
# )

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

# custom_hooks = [
#     dict(type='OccEfficiencyHook'),
# ]
