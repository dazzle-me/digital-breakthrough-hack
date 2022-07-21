_base_ = '/workspace/rlh/mmsegmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

# stride_size = 90
# h_dim = 1440
# w_dim = 2560

# assert h_dim % stride_size == 0

# crop_size = (w_dim, stride_size)
# img_scale = (w_dim, h_dim)


crop_size = (1024, 1024)
stride = (768, 768)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=4),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

train_image_dir = 'train/images'
test_image_dir = 'test'
ann_dir = 'train/new_mask'

data_root = '/workspace/datasets/rlh/data'
dist_params = {
    'backend' : 'nccl'
}

# optimizer = dict(
#     type='AdamW',
#     lr=6e-05,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             absolute_pos_embed=dict(decay_mult=0.0),
#             relative_position_bias_table=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0)
#         )
#     )
# )

# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=load_from),
#         pretrain_img_size=384,
#         embed_dims=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=12),
#     decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=4),
#     auxiliary_head=dict(in_channels=512, num_classes=4))

# model = dict(
#     backbone=dict(
#         type='mmcls.ConvNeXt',
#         arch='large',
#         out_indices=[0, 1, 2, 3],
#         drop_path_rate=0.4,
#         layer_scale_init_value=1.0,
#         gap_before_final_norm=False,
#         init_cfg=dict(
#             type='Pretrained', checkpoint=load_from,
#             prefix='backbone.')),
#     decode_head=dict(
#         in_channels=[192, 384, 768, 1536],
#         num_classes=4,
#     ),
#     auxiliary_head=dict(in_channels=768, num_classes=4),
#     test_cfg=dict(
#         # mode='slide', crop_size=crop_size, stride=(426, 426)
#         mode='whole'
#     ),
# )
# norm_cfg = dict(type='BN', requires_grad=True)
# model = dict(
#     backbone=dict(
#         pretrained=None,
#         norm_cfg=norm_cfg,
        # extra=dict(
        #     stage1=dict(num_blocks=(2, )),
        #     stage2=dict(num_blocks=(2, 2)),
        #     stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
        #     stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2))),
        # decode_head=dict(
        #     norm_cfg=norm_cfg,
        #     num_classes=4
        # ))
# """Since the given config is used to train PSPNet on the cityscapes dataset, we need to modify it accordingly for our new dataset.  """

# Since we use only one GPU, BN is used instead of SyncBN
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# ## cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
# cfg.model.decode_head.num_classes = 4
## cfg.model.auxiliary_head.num_classes = 4

# Modify dataset type and path


# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     warmup='linear',
#     warmup_iters=100,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5,
#     min_lr=1e-6
# )

# optimizer = dict(
#     type='AdamW',
#     lr=6e-05,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             absolute_pos_embed=dict(decay_mult=0.0),
#             relative_position_bias_table=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0)
#         )
#     )
# )
# optimizer_config = dict()

dataset_type = 'RailsDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=img_scale, ratio_range=(1.0, 1.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomFlip'),
#     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img']),
# ]


# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_image_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split='stratified_split/train.txt'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_image_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='stratified_split/val.txt'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=test_image_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='stratified_split/test.txt'
    )
)

runner = dict(
    max_iters=320_000
)
log_config = dict(
    interval=25
)
evaluation = dict(
    interval=8000,
    metric='mIoU',
    save_best='mIoU'
)
checkpoint_config = dict(
    interval=999999999
)
# Set seed to facitate reproducing the result
seed = 0
cudnn_benchmark = True

# set_random_seed(0, deterministic=False)
gpu_ids = range(2)
device='cuda'
