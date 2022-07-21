# _base_ = 'mmsegmentation/configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-d24c28c1.pth'

train_image_dir = 'train/images'
test_image_dir = 'test'
ann_dir = 'train/new_mask'

data_root = '/workspace/datasets/rlh/data'
work_dir = '/workspace/rlh/work_dirs/exp4'
dist_params = {
    'backend' : 'nccl'
}

norm_cfg = dict(type='BN', requires_grad=True)
_base_ = 'mmsegmentation/configs/fcn/fcn_r101-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=0.125,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(in_channels=40),
    auxiliary_head=dict(in_channels=16))

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# # fp16 placeholder
# fp16 = dict()

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
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

img_scale=(2048, 1024)
crop_size = (512, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.85, 1.15)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    # dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
work_dir = work_dir

runner = dict(
    max_iters=80000
)
log_config = dict(
    interval=25
)
evaluation = dict(
    interval=500,
    metric='mIoU'
)
save_best='mIoU'
checkpoint_config = dict(
    interval=500
)
# Set seed to facitate reproducing the result
seed = 0
cudnn_benchmark = True

# set_random_seed(0, deterministic=False)
gpu_ids = [0]
device='cuda'
