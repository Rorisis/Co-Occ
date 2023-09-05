_base_ = [
    '../_base_/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

sync_bn = True
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['empty', 'barrier', 'bicycle', 'bus', 'car', 
    'construction_vehicle', 'motorcycle', 'pedestrian', 
    'traffic_cone', 'trailer', 'truck', 'driveable_surface', 
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
num_class = len(class_names)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 32]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channel = 256
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False

# settings for mask2former head
# mask2former_num_queries = 100
# mask2former_feat_channel = voxel_out_channels
# mask2former_output_channel = voxel_out_channels
# mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
# mask2former_num_heads = voxel_out_channels // 32

model = dict(
    type='MoEOccpancy',
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
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=numC_Trans,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024, 1024, 128],  # hardcode, xy size follow centerpoint
        ),
    occ_fuser=dict(
        type='VisFuser',
        in_channels=numC_Trans,
        out_channels=numC_Trans,
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
        fine_topk=15000,
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
#         # loss settings
#         loss_cls=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=2.0,
#             reduction='mean',
#             class_weight=[1.0] * num_class + [0.1]),
#         loss_mask=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=5.0),
#         loss_dice=dict(
#             type='DiceLoss',
#             use_sigmoid=True,
#             activate=True,
#             reduction='mean',
#             naive_dice=True,
#             eps=1.0,
#             loss_weight=5.0),
#         point_cloud_range=point_cloud_range,
#     ),
#     train_cfg=dict(
#         pts=dict(
#             num_points=12544 * 4,
#             oversample_ratio=3.0,
#             importance_sample_ratio=0.75,
#             assigner=dict(
#                 type='MaskHungarianAssigner',
#                 cls_cost=dict(type='ClassificationCost', weight=2.0),
#                 mask_cost=dict(
#                     type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
#                 dice_cost=dict(
#                     type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
#             sampler=dict(type='MaskPseudoSampler'),
#         )),
#     test_cfg=dict(
#         pts=dict(
#             semantic_on=True,
#             panoptic_on=False,
#             instance_on=False)),
# )

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes'
nusc_class_metas = 'projects/configs/_base_/nuscenes.yaml'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,)

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
    dict(type='LoadNuscOccupancyAnnotations', is_train=True, grid_size=occ_size, 
            point_cloud_range=point_cloud_range, bda_aug_conf=bda_aug_conf,
            cls_metas=nusc_class_metas),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
            meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_OccFormer', is_train=False,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='LoadNuscOccupancyAnnotations', is_train=False, grid_size=occ_size,
            point_cloud_range=point_cloud_range, bda_aug_conf=bda_aug_conf,
            cls_metas=nusc_class_metas),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points', 'points_occ'],
            meta_keys=['pc_range', 'occ_size', 'sample_idx', 'timestamp',
                       'scene_token', 'img_filenames', 'scene_name']),
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

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
    samples_per_gpu=2,
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

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        },
        norm_decay_mult=0.0))

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[20, 23],
)

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=24)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='nuScenes_lidarseg_mean',
    rule='greater',
)