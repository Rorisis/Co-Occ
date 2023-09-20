point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_OccFormer',
        is_train=False,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    dict(type='CreateDepthFromLiDAR', dataset='nusc'),
    dict(
        type='LoadNuscOccupancyAnnotations',
        is_train=True,
        grid_size=[256, 256, 32],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0.5),
        cls_metas='projects/configs/_base_/nuscenes.yaml'),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
        meta_keys=['pc_range', 'occ_size'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_OccFormer',
        is_train=False,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    dict(
        type='LoadNuscOccupancyAnnotations',
        is_train=False,
        grid_size=[256, 256, 32],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0.5),
        cls_metas='projects/configs/_base_/nuscenes.yaml'),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ],
        with_label=False),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
        meta_keys=[
            'pc_range', 'occ_size', 'sample_idx', 'timestamp', 'scene_token',
            'img_filenames', 'scene_name'
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesOccLSSDataset',
        data_root='data/nuscenes',
        ann_file='data/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
            dict(
                type='LoadMultiViewImageFromFiles_OccFormer',
                is_train=False,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(type='CreateDepthFromLiDAR', dataset='nusc'),
            dict(
                type='LoadNuscOccupancyAnnotations',
                is_train=True,
                grid_size=[256, 256, 32],
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                cls_metas='projects/configs/_base_/nuscenes.yaml'),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'empty', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ]),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
                meta_keys=['pc_range', 'occ_size'])
        ],
        classes=[
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        occ_size=[256, 256, 32],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    val=dict(
        type='CustomNuScenesOccLSSDataset',
        ann_file='data/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
            dict(
                type='LoadMultiViewImageFromFiles_OccFormer',
                is_train=False,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(
                type='LoadNuscOccupancyAnnotations',
                is_train=False,
                grid_size=[256, 256, 32],
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                cls_metas='projects/configs/_base_/nuscenes.yaml'),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'empty', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sample_idx', 'timestamp',
                    'scene_token', 'img_filenames', 'scene_name'
                ])
        ],
        classes=[
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/nuscenes',
        occ_size=[256, 256, 32],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    test=dict(
        type='CustomNuScenesOccLSSDataset',
        data_root='data/nuscenes',
        ann_file='data/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
            dict(
                type='LoadMultiViewImageFromFiles_OccFormer',
                is_train=False,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(
                type='LoadNuscOccupancyAnnotations',
                is_train=False,
                grid_size=[256, 256, 32],
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                cls_metas='projects/configs/_base_/nuscenes.yaml'),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'empty', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sample_idx', 'timestamp',
                    'scene_token', 'img_filenames', 'scene_name'
                ])
        ],
        classes=[
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_size=[256, 256, 32],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
        dict(
            type='LoadMultiViewImageFromFiles_OccFormer',
            is_train=False,
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),
        dict(
            type='LoadNuscOccupancyAnnotations',
            is_train=False,
            grid_size=[256, 256, 32],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0.5),
            cls_metas='projects/configs/_base_/nuscenes.yaml'),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'empty', 'barrier', 'bicycle', 'bus', 'car',
                'construction_vehicle', 'motorcycle', 'pedestrian',
                'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'sample_idx', 'timestamp',
                'scene_token', 'img_filenames', 'scene_name'
            ])
    ],
    save_best='nuScenes_lidarseg_mean',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_ne_nusc_multi'
load_from = None
resume_from = None
workflow = [('train', 1)]
sync_bn = True
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_class = 17
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]
voxel_x = 0.4
voxel_y = 0.4
voxel_z = 0.25
voxel_size = [0.4, 0.4, 0.25]
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-5.0, 3.0, 0.5],
    dbound=[2.0, 58.0, 0.5])
numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channel = 256
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
empty_idx = 0
num_cls = 17
visible_mask = False
cascade_ratio = 2
sample_from_voxel = True
sample_from_img = True
model = dict(
    type='NeRFOcc',
    loss_norm=True,
    voxel_size=[0.4, 0.4, 0.25],
    n_voxels=[256, 256, 32],
    aabb=([-51.2, -51.2, -5.0], [51.2, 51.2, 3.0]),
    near_far_range=[0.2, 50.0],
    N_samples=64,
    N_rand=2048,
    depth_supervise=True,
    use_nerf_mask=True,
    nerf_sample_view=6,
    squeeze_scale=4,
    nerf_density=True,
    use_rendering=False,
    loss_voxel_ce_weight=1.0,
    loss_voxel_sem_scal_weight=1.0,
    loss_voxel_geo_scal_weight=1.0,
    loss_voxel_lovasz_weight=1.0,
    img_backbone=dict(
        pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        loss_depth_weight=1.0,
        loss_depth_type='bce',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-5.0, 3.0, 0.5],
            dbound=[2.0, 58.0, 0.5]),
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        numC_Trans=128,
        vp_megvii=False),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.1, 0.1, 0.1],
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=128,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024, 1024, 128]),
    occ_fuser=dict(type='VisFuser', in_channels=128, out_channels=128),
    density_encoder=dict(
        type='CustomResNet3D',
        depth=10,
        n_input_channels=128,
        block_inplanes=[128, 256, 512, 1024],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    color_encoder=dict(
        type='CustomResNet3D',
        depth=10,
        n_input_channels=128,
        block_inplanes=[128, 256, 512, 1024],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    semantic_encoder=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=2,
        sample_from_voxel=True,
        sample_from_img=True,
        final_occ_size=[256, 256, 32],
        fine_topk=10000,
        empty_idx=0,
        num_level=4,
        in_channels=[256, 256, 256, 256],
        out_channel=17,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0)),
    empty_idx=0)
nusc_class_metas = 'projects/configs/_base_/nuscenes.yaml'
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5)
test_config = dict(
    type='CustomNuScenesOccLSSDataset',
    data_root='data/nuscenes',
    ann_file='data/nuscenes_infos_temporal_val.pkl',
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
        dict(
            type='LoadMultiViewImageFromFiles_OccFormer',
            is_train=False,
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),
        dict(
            type='LoadNuscOccupancyAnnotations',
            is_train=False,
            grid_size=[256, 256, 32],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0.5),
            cls_metas='projects/configs/_base_/nuscenes.yaml'),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'empty', 'barrier', 'bicycle', 'bus', 'car',
                'construction_vehicle', 'motorcycle', 'pedestrian',
                'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'sample_idx', 'timestamp',
                'scene_token', 'img_filenames', 'scene_name'
            ])
    ],
    classes=[
        'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'
    ],
    modality=dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    occ_size=[256, 256, 32],
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy='step', step=[20, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
gpu_ids = range(0, 2)
