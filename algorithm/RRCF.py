# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import RobustRandomCutForest
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = RobustRandomCutForest(**config['argument'])
    auroc = AUROCMetric()
    all_scores = []
    all_labels = []
    start_cpu_time = time.process_time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, score)
        all_scores.append(score)
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
              f'num_trees: {config["argument"]["num_trees"]}\n'
              f'shingle_size: {config["argument"]["shingle_size"]}\n'
              f'tree_size: {config["argument"]["tree_size"]}\n',
              end='\n\n',
              file=f)
        
def run_rrcf_detector(data, label, config_args):
    """
    封装 RRCF 的运行过程，返回所有样本的异常分数序列
    """
    from pysad.models import RobustRandomCutForest
    from pysad.utils import ArrayStreamer
    import numpy as np

    # 1. 初始化模型 (config_args 通常包含 n_estimators, tree_size 等)
    model = RobustRandomCutForest(**config_args)
    
    # 2. 设置流式迭代器
    iterator = ArrayStreamer(shuffle=False)
    all_scores = []
    
    # 3. 模拟流式处理
    for x, _ in iterator.iter(data, label):
        # fit_score_partial 在 RRCF 中计算位移量(displacement)作为异常得分
        score = model.fit_score_partial(x)
        all_scores.append(score)
        
    return np.array(all_scores).flatten()