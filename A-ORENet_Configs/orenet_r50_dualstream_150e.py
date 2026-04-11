custom_imports = dict(imports=['orenet_modules.custom_head'], allow_failed_imports=False)

_base_ = [
    '../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

# 环境增强：防止多线程死锁
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

dataset_type = 'CocoDataset'
classes = ('stone',)

# 1. 训练配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='/group/chenjinming/Datas/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')
    )
)

# 2. 验证配置 (关键修复：num_workers=0 彻底杜绝死锁)
val_dataloader = dict(
    batch_size=1,
    num_workers=0, 
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='/group/chenjinming/Datas/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/group/chenjinming/Datas/annotations/instances_val2017.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# 3. 模型配置 (ResNet50 + 双流头)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            _delete_=True,
            type='DualStreamDecoupledHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            loss_edge=dict(type='SpatialGradientLoss', loss_weight=0.05) 
        )
    )
)

# 4. 训练策略
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2) 
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), 
    dict(type='MultiStepLR', begin=0, end=150, by_epoch=True, milestones=[100, 130], gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=10)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
work_dir = 'A-ORENet_Outputs/weights'
