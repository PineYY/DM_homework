checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=80,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO' #DEBUG,INFO,WARNING,ERROR,CRITICAL
#load_from = '/home/yanghanfang/mmdetection/mask_rcnn_r50_fpn_1x_3.pth'
#load_from = '/home/yanghanfang/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
load_from = ''
resume_from = None
workflow = [('train', 1)]
