# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import HalfSpaceTrees
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

# 在 HSTree.py 中添加此函数
def run_hstree_detector(data, label, config_args):
    """
    封装 HSTree 的运行过程，返回所有样本的异常分数
    """
    from pysad.models import HalfSpaceTrees
    from pysad.utils import ArrayStreamer
    
    # 初始化模型
    model = HalfSpaceTrees(**dict({
        'feature_mins': np.array(np.min(data)).reshape(-1),
        'feature_maxes': np.array(np.max(data)).reshape(-1)
    }, **config_args))
    
    iterator = ArrayStreamer(shuffle=False)
    all_scores = []
    
    # 流式处理获取分数
    for x, _ in iterator.iter(data, label):
        score = model.fit_score_partial(x)
        all_scores.append(score)
        
    return np.array(all_scores)

def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = HalfSpaceTrees(**dict({'feature_mins': np.array(np.min(data)).reshape(-1),
                                   'feature_maxes': np.array(np.max(data)).reshape(-1)}, **config['argument']))
    auroc = AUROCMetric()
    all_labels = []
    all_scores = []
    start_cpu_time = time.process_time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, score)
        all_labels.append(y)
        all_scores.append(score)

    # 计算新指标
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    # 以HalfSpaceTrees为例，在生成predictions前添加：
    all_scores = np.array(all_scores).flatten()  # 算法输出的异常分数
    all_labels = np.array(all_labels).flatten()  # 真实标签

# 计算动态阈值（推荐用ROC最优阈值，兼顾精确率和召回率）
    # --- 自定义范围切片 ---
    eval_start, eval_end = config.get('auc_range', [1, len(all_labels)])
    sliced_labels = all_labels[eval_start-1 : eval_end]
    sliced_scores = all_scores[eval_start-1 : eval_end]

    # 使用切片后的数据计算指标
    fpr_roc, tpr_roc, thresholds = roc_curve(sliced_labels, sliced_scores)
    optimal_idx = np.argmax(tpr_roc - fpr_roc)
    threshold = thresholds[optimal_idx]
    
    roc_auc = roc_auc_score(sliced_labels, sliced_scores)
    predictions = (sliced_scores > threshold).astype(int)
    precision = precision_score(sliced_labels, predictions, zero_division=0)
    recall = recall_score(sliced_labels, predictions, zero_division=0)
    f1 = f1_score(sliced_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(sliced_labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_cpu_time = time.process_time() - start_cpu_time

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
              f'window_size: {config["argument"]["window_size"]}\n'
              f'num_trees: {config["argument"]["num_trees"]}\n'
              f'max_depth: {config["argument"]["max_depth"]}\n',
              end='\n\n',
              file=f)

# 在 HSTree.py 中添加此函数
def run_hstree_detector(data, label, config_args):
    """
    封装 HSTree 的运行过程，返回所有样本的异常分数
    """
    from pysad.models import HalfSpaceTrees
    from pysad.utils import ArrayStreamer
    
    # 初始化模型
    model = HalfSpaceTrees(**dict({
        'feature_mins': np.array(np.min(data)).reshape(-1),
        'feature_maxes': np.array(np.max(data)).reshape(-1)
    }, **config_args))
    
    iterator = ArrayStreamer(shuffle=False)
    all_scores = []
    
    # 流式处理获取分数
    for x, _ in iterator.iter(data, label):
        score = model.fit_score_partial(x)
        all_scores.append(score)
        
    return np.array(all_scores)