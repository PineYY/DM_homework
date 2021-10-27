# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='Adam', lr=0.003, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[10,],
    gamma=1/3)
runner = dict(type='EpochBasedRunner', max_epochs=1000)
