# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmaction.registry import METRICS
from .acc_metric import AccMetric

@METRICS.register_module()
class DualAccMetric(AccMetric):
    """專門為上下半身雙輸出設計的 Accuracy 評估指標"""
    """Accuracy evaluation metric."""
    """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
    """
    
    default_prefix = 'dual_acc'

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            label = data_sample['gt_label']
            label_debug = label.reshape(-1)

            if label_debug.numel() != 2:
                raise ValueError(
                    f'[DualAccMetric] gt_label 形狀異常: '
                    f'shape={tuple(label.shape)}, value={label}'
                )

            gt_upper = label[0].item()
            gt_lower = label[1].item()
            # 抓出預測分數 (我們在 DualBaseHead 裡塞進去的屬性)
            pred_upper = data_sample.get('pred_score_upper')
            pred_lower = data_sample.get('pred_score_lower')
            # 應該是原本的 pred = data_sample['pred_score']
            
            if pred_upper is None or pred_lower is None:
                raise KeyError(
                    f'[DualAccMetric] 找不到 pred_score_upper/lower，'
                    f'可用欄位: {list(data_sample.keys())}')

            if not isinstance(pred_upper, np.ndarray):
                pred_upper = pred_upper.cpu().numpy()
            if not isinstance(pred_lower, np.ndarray):
                pred_lower = pred_lower.cpu().numpy()

            # 目前 config 是 upper=2 類, lower=7 類
            """
            if pred_upper_debug.shape[0] != 7:
                raise ValueError(f'[DualAccMetric] pred_upper 維度錯誤: got {pred_upper_debug.shape}, 預期 (2,)')

            if pred_lower_debug.shape[0] != 2:
                raise ValueError(f'[DualAccMetric] pred_lower 維度錯誤: got {pred_lower_debug.shape}, 預期 (7,)')
            """

            result['pred_upper'] = pred_upper
            result['pred_lower'] = pred_lower
            result['gt_upper'] = gt_upper
            result['gt_lower'] = gt_lower
            
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # 分流計算並合併結果
        #labels = [x['label'] for x in results]

        eval_results = dict()
        # 上下半身都只算 top-1
        self.metric_options['top_k_accuracy'] = dict(topk=(1, ))
        """
        acc_metric.py 說
        topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
        """
        preds_upper = [x['pred_upper'] for x in results]
        labels_upper = [x['gt_upper'] for x in results]
        # 呼叫老爹的 calculate，它會自動幫我們算 top-1, top-5...
        upper_metrics = self.calculate(preds_upper, labels_upper)
        # 在輸出的指標名稱前面加上 'upper_' 以作區別

        # === 計算下半身 (Lower Body) 指標 ===
        preds_lower = [x['pred_lower'] for x in results]
        labels_lower = [x['gt_lower'] for x in results]
        lower_metrics = self.calculate(preds_lower, labels_lower)

        # === 綜合指標放最前面 ===
        # 用 mean1 (per-class recall 平均) 而非 top1，避免被多數類灌水
        if 'mean1' in upper_metrics and 'mean1' in lower_metrics:
            eval_results['mean1'] = (upper_metrics['mean1'] +
                                     lower_metrics['mean1']) / 2
        if 'top1' in upper_metrics and 'top1' in lower_metrics:
            eval_results['top1'] = (upper_metrics['top1'] +
                                    lower_metrics['top1']) / 2

        for k, v in upper_metrics.items():
            eval_results[f'upper_{k}'] = v
        
        # 在輸出的指標名稱前面加上 'lower_' 以作區別
        for k, v in lower_metrics.items():
            eval_results[f'lower_{k}'] = v

        return eval_results

