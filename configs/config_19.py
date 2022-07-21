_base_ = 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k.py'
load_from = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'

train_image_dir = 'train/images'
test_image_dir = 'test'
ann_dir = 'train/new_mask'

data_root = '/workspace/datasets/rlh/data'
dist_params = {
    'backend' : 'nccl'
}
crop_size = (640, 640)
img_scale = crop_size


model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=load_from,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=4,
    ),
    auxiliary_head=dict(in_channels=768, num_classes=4),
    test_cfg=dict(
        # mode='slide', crop_size=crop_size, stride=(426, 426)
        mode='whole'
    ),
)
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
dataset_type = 'RailsDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(1.0, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomFlip'),
#     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img']),
# ]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_image_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split='splits/train.txt'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_image_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='splits/val.txt'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=test_image_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='splits/test.txt'
    )
)

runner = dict(
    max_iters=80_000
)
log_config = dict(
    interval=25
)
evaluation = dict(
    interval=500,
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
gpu_ids = [0]
device='cuda'
