# 訓練/驗證/測試時，整個 runner 系統怎麼運作
# 例如：多久 log 一次、多久存 checkpoint、要不要續訓、分散式環境怎麼設。
default_scope = 'mmaction' # 預設 mmaction 為預設路徑，type=... 都要靠它來尋找目的地

# hooks: 鉤子
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 好像是給 linux 的，我的電腦是 windows，不知道會有甚麼問題
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = "configs/skeleton/posec3d/checkpoints/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth"
#　原本預設不從某個 checkpoint 載入權重。 遷移學習 要改這個
resume = False # 不要從之前的訓練狀態繼續訓練，完全重新開始訓練。
