_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-40, 0, -40, 40, 80, 40]
point_cloud_range = [-72, -72, -104, 72, 72, 8]
# occ_size = [80, 80, 80]
occ_size = [72, 72, 56]

use_semantic = True


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['Terrain','Vegetation','Water','Bridge','Vehicle','Boat','Building']


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [36, 18, 9]
volume_w_ = [36, 18, 9]
volume_z_ = [28, 14, 7]
# volume_h_ = [40, 20, 10]
# volume_w_ = [40, 20, 10]
# volume_z_ = [40, 20, 10]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    is_vis=True,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=8,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_cams=5,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=5,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
),
)

dataset_type = 'CustomUrbOccDataset'
data_root = '/media/vcc/LinuxData/project/data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/mnt/d/project/SurroundOcc-main/data/train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        pkl_file_path = "/mnt/d/project/SurroundOcc-main/data/train.pkl",
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             # ann_file='data/nuscenes_infos_val.pkl',
             ann_file="/mnt/d/project/SurroundOcc-main/data/val.pkl",
             pipeline=test_pipeline,
             occ_size=occ_size,
             pc_range=point_cloud_range,
             use_semantic=use_semantic,
             classes=class_names,
             pkl_file_path = "/mnt/d/project/SurroundOcc-main/data/val.pkl",
             modality=input_modality),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file="/mnt/d/project/SurroundOcc-main/data/train.pkl",
              # ann_file='data/inference_infos.pkl',
              pipeline=test_pipeline,
              occ_size=occ_size,
              pc_range=point_cloud_range,
              use_semantic=use_semantic,
              classes=class_names,
              pkl_file_path = "/mnt/d/project/SurroundOcc-main/data/train.pkl",
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
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
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
