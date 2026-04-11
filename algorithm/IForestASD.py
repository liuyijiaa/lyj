# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import IForestASD
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score

def main(config):
    warnings.filterwarnings("ignore")
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = IForestASD(**config['argument'])
    auroc = AUROCMetric()
    all_scores = []
    all_labels = []
    start_cpu_time = time.process_time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, score)
        all_scores.append(score)  # 收集分数
        all_labels.append(y)
    total_cpu_time = time.process_time() - start_cpu_time
    all_scores = np.array(all_scores).flatten()
    all_labels = np.array(all_labels).flatten()

    # --- 自定义范围切片 ---
    eval_start, eval_end = config.get('auc_range', [1, len(all_labels)])
    sliced_labels = all_labels[eval_start-1 : eval_end]
    sliced_scores = all_scores[eval_start-1 : eval_end]

    roc_auc = roc_auc_score(sliced_labels, sliced_scores)
    predictions = (sliced_scores > 0).astype(int)
    precision = precision_score(sliced_labels, predictions, zero_division=0)
    recall = recall_score(sliced_labels, predictions, zero_division=0)
    f1 = f1_score(sliced_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(sliced_labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'Evaluation Range: {eval_start} to {eval_end}\n'
              f'ROC: {roc_auc:.4f}\n'
              f'Precision: {precision:.4f}\n'
              f'Recall: {recall:.4f}\n'
              f'F1: {f1:.4f}\n' 
              f'False Positive Rate (FPR): {fpr:.4f}\n'
              f'CPU Time: {total_cpu_time:.4f} seconds\n'
              f'window_size: {config["argument"]["window_size"]}\n',
              end='\n\n',
              file=f)

def run_iforest_detector(data, label, config_args):
    """
    为可视化脚本提供的封装接口
    """
    from pysad.models import IForestASD
    from pysad.utils import ArrayStreamer
    import numpy as np

    # 1. 初始化模型 (使用传入的配置参数)
    model = IForestASD(**config_args)
    
    # 2. 设置迭代器
    iterator = ArrayStreamer(shuffle=False)
    all_scores = []
    
    # 3. 模拟流式处理
    for x, _ in iterator.iter(data, label):
        # fit_score_partial 同时训练并返回当前样本的异常分数
        score = model.fit_score_partial(x)
        all_scores.append(score)
        
    return np.array(all_scores).flatten()