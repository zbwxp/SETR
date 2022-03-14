_base_ = [
    '../../_base_/models/setr_naive_pup.py',
    '../../_base_/datasets/ade20k.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 768
model = dict(
    backbone=dict(
        model_name='vit_base_patch16_384',
        type="vit_2x2_v2",
        depth=12,
        embed_dim=in_channels,
        num_heads=12,
        img_size=img_size,
        align_corners=False,
        pos_embed_interp=True,
        drop_rate=0.,
        num_classes=150),
    decode_head=dict(
        type='TPN_merge',
        in_channels=in_channels,
        embed_dim=in_channels // 2,
        reverse=True,
        num_expand_layer=3,
        use_norm=True,
        in_index=11,
        img_size=img_size,
        align_corners=False,
        num_conv=2,
        upsampling_method='bilinear',
        num_classes=150,
        conv3x3_conv1x1=False,
        norm_cfg=norm_cfg, ),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=in_channels,
        channels=512,
        in_index=5,
        img_size=img_size,
        embed_dim=in_channels,
        num_classes=150,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='up_from_2x2',
            expand_query=256,
            in_channels=in_channels,
            embed_dim=in_channels // 4,
            channels=512,
            in_index=7,
            img_size=img_size,
            align_corners=False,
            num_conv=2,
            upsampling_method='bilinear',
            num_classes=150,
            conv3x3_conv1x1=False,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
        dict(
            type='up_from_2x2',
            expand_query=256,
            in_channels=in_channels,
            embed_dim=in_channels // 4,
            channels=512,
            in_index=9,
            img_size=img_size,
            align_corners=False,
            num_conv=2,
            upsampling_method='bilinear',
            num_classes=150,
            conv3x3_conv1x1=False,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
    ])

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 }))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

crop_size = (img_size, img_size)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
find_unused_parameters = True
data = dict(samples_per_gpu=2,
            workers_per_gpu=2)
evaluation = dict(interval=4000, metric='mIoU')
