load_from = 'checkpoints/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth'
custom_imports = dict( # key: 'imports', value 後面的東西
    imports=[
        'mmaction.datasets.dual_pose_dataset',  
        'mmaction.models.heads.multi_i3d_head', 
        'mmaction.evaluation.metrics.dual_acc_metric', 
        'mmaction.datasets.transforms.SG_filter', 
        'mmaction.models.recognizers.custom_recognizer',
        'mmaction.models.necks.dual_window_gating'
    ],
    allow_failed_imports=False)
# 這裡在找位置
# 簡單講就是不管有沒有用到都先載入我自己準備的檔案

_base_ = '../../_base_/default_runtime.py' # 這份 config 繼承另一份基底設定檔。

"""
這個 model 的 backbone 有 3 個 stages，
每個 stage 由 多個 Residual Blocks 組成，
最後只把 stage2（從 0 開始數） 的輸出特徵圖拿出來給後面的 head。
"""
# 推論會用到
model = dict(
    type='CustomDualRecognizer', # 原 Recognizer3D
    backbone=dict(
        type='ResNet3dSlowOnly', # 用 3D ResNet 架構 (時間 + 高度 + 寬度 (T, H, W)) 
        depth=50,
        pretrained=None, # 未來遷移訓練這部分可能要改
        in_channels=17, # backbone 輸入通道數是 17 個關鍵點 
        base_channels=32,
        num_stages=3,
        out_indices=(1, 2), # 現在改成輸出 layer 2 和 layer 3，分別時解析度與全域視野(就是那個神經網路的 p) 
        stage_blocks=(4, 6, 3), #　每個　stage 幾個 blocks，blocks = Residual Blocks
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2), #　會影響 T（時間長度）有沒有被下採樣　看起來是有（最後一個）
        dilations=(1, 1, 1)),
    neck=dict( #　這是 backbone 和 head 的中間區塊
        type='DualWindowGatingNeck',
        in_channels=512,    # ResNet50 stage 3 的輸出通道數通常是 512。預期 backbone 傳進來的 feature channel 是 512。 可以確認 backbone output.shape
        l2_in_channels=256,   # 新增：layer2 的輸出通道數
        sg_feat_dim=340,    # 記得確認 S-G 特徵真實維度 (大視窗單人 136 + 小視窗單人 136) (後來又多加了靜位移與方向一致性)看 SG_filter.py 的實作。
        out_channels=512,
        crop_margin=3, 
        debug=True          # 開啟 debug 方便確認維度
    ),
    cls_head=dict(
        type='DualI3DHead', # 改
        debug=False,        # （想開除錯時改 True）
        in_channels=512, # 表示 head 預期從 neck 收到的特徵 channel 是 512。
        num_classes_upper=2, # 上半身標籤
        num_classes_lower=6, # 下
        dropout_ratio=0.5,
        average_clips='prob',
        topk=(1, ),                       # 新增：關掉訓練 log 的假 top5
        loss_weight_upper=0.5,            # 新增
        loss_weight_lower=1.0,            # 新增
        loss_cls_upper=dict(              # 新增
            type='CrossEntropyLoss',
            class_weight=[1.00, 1.48]),
        loss_cls_lower=dict(              # 新增
            type='CrossEntropyLoss',
            class_weight=[1.58, 1.27, 1.00, 7.26, 5.79, 2.28]))) # 不是先平均 logits 再 softmax，而是先變成 probability 再平均
            # 或 [1.26, 1.13, 1.00, 2.69, 2.41, 1.51] 當消融
"""
所以整條鏈目前是：

backbone 輸出：512 ch 接 neck

neck 處理後：仍是 512 ch 接 head

head 輸入：512 ch
可以再做確認

gt_label = [上半身標籤, 下半身標籤]

"""
# 用哪種資料集格式
# 標註檔在哪
dataset_type = 'DualPoseDataset' # 改 原 PoseDataset'PoseDataset'
# DualPoseDataset 期待的標註格式，必須真的和我準備的 pkl 裡的每筆資料相容。
ann_file = 'pkl/train_val.pkl' # 遷移訓練時要注意 要更動
""" 目前這些我用不到，因為我是自己準備轉 PKL 檔案。 """

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

"""
現在實際推論時，模型看到的是先切好的 window_size 視窗；
test_pipeline 再從這個視窗內抽 21 幀、抽 num_clips 組。
15 幀短視窗則是從 21 幀長視窗內再裁出來的。這樣比較快。
"""
# 定義了訓練時一筆 skeleton 資料要怎麼被抽樣、增強、轉成 heatmap
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=21), # 抽固定長度片段（這裡改成長視窗的 21 幀） 
    dict(type='PoseDecode'), #　把對應幀的骨架資料取出來
    # window_length: 在時間序列上，看「附近幾幀」的資料，做平滑，減少抖動。
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # PoseCompact: 把骨架座標「重新框到更緊的人體區域」，讓後面產生 heatmap 時不要浪費太多空白背景。
    # 以上都在 pose_transforms.py
    dict(type='Resize', scale=(-1, 64)),
    # backbone 吃的是 heatmap，最終會變成 (N, C, T, H, W)，可以預期後面樣子應該會是 (batch, 17, T, 56, 56)
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp), #會翻轉，有危險性
    dict(type='SG', crop_margin=3, window_length=5, polyorder=2), # SG 濾波器
    # crop_margin: 從長視窗的前後各裁掉 3 幀，得到短視窗。
    # window_length: S-G 濾波器的滑動視窗大小(一次看多少來濾)。數字越大，濾波越平滑但細節越少。必須是奇數。
    # polyorder: 2 表示用二次多項式(除了速度還有加速度)
    # 大小視窗跟這裡也有關
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),

    # 轉成 heatmap，原本是 keypoint: (M, T, 17, 2) 要轉成 HEATMAP
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'), # 這一步把資料整理成 backbone 要吃的 5 維格式 (N, C, T, H, W)。
    #dict(type='PackActionInputs'), 原
    dict(type='PackActionInputs', meta_keys=('img_shape', 'ori_shape', 'frame_dir', 'clip_start'), algorithm_keys=('sg_features',))
    # PackActionInputs 在 formatting.py
    # sg_features 應該會在 results['sg_features'] = sg_features.astype(np.float32)，SG_FILTER.PY 包裝好
]
"""
meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
"""
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=21, num_clips=1, test_mode=True), 
    # num_clips： 從同一段影片/骨架序列中，取幾個 clip（片段) 來做推論或驗證。
    # clip_len 是每組 clip 要幾幀
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='SG', crop_margin=3, window_length=5, polyorder=2),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    #dict(type='PackActionInputs'),
    dict(type='PackActionInputs', meta_keys=('img_shape', 'ori_shape', 'frame_dir', 'clip_start'), algorithm_keys=('sg_features',))
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=21, num_clips=1, test_mode=True), # 原 num_clips=10
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='SG', crop_margin=3, window_length=5, polyorder=2),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False, # 接下來要做「雙視窗」，而雙視窗本來就會讓 input shape 變更複雜。考慮改成 false
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    #dict(type='PackActionInputs'),
    dict(type='PackActionInputs', meta_keys=('img_shape', 'ori_shape', 'frame_dir', 'clip_start'), algorithm_keys=('sg_features',))
]
# 這段就是決定訓練資料如何進模型。
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2, #若原本 100 筆訓練資料：times=10 後，epoch 內等效成 1000 筆
        # times=10 可能讓訓練太長、過擬合更快。
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='xsub_train', # 所以 pkl 一定要以 xsub_train 為頭，可以參考 https://blog.csdn.net/m0_59670748/article/details/139546804 這篇
            window_len=21,                # 新增
            window_stride=1,              # 新增：訓練用密集滑窗，涵蓋所有相位
            max_windows=5,
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # DataLoader 每次要按什麼順序取資料 index, true 打亂, false 不打亂
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        window_len=21,
        window_stride=1,
        max_windows=5,
        pipeline=val_pipeline,
        test_mode=True))

# 不直接用它。
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        window_len=21,
        window_stride=21,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='DualAccMetric')] # 改 原 val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=24, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 未來做遷移訓練時，這是很常被改的一段，尤其當資料量比 NTU60 小很多時。
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=24,
        by_epoch=True,
        convert_to_iter_based=True)
]
# 遷移訓練會用到
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16', 
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0003),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.05),  # 實際 lr = 0.002,backbone 慢慢微調
        'neck':     dict(lr_mult=1.0),   # 實際 lr = 0.04,從零開始學
        'cls_head': dict(lr_mult=1.0),   # 實際 lr = 0.04,從零開始學
    }),
    clip_grad=dict(max_norm=40, norm_type=2))

randomness = dict(seed=42, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best=[
            'dual_acc/mean1',        # 綜合最好
            'dual_acc/upper_mean1',   # 上半身最好
            'dual_acc/lower_mean1',  # 下半身最好
        ],
        rule=['greater', 'greater', 'greater']))
