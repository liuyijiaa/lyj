# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import RSHash
from pysad.utils import ArrayStreamer
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
    model = RSHash(**dict({'feature_mins': np.array(np.min(data)).reshape(-1),
                           'feature_maxes': np.array(np.max(data)).reshape(-1)}, **config['argument']))
    auroc = AUROCMetric()
    all_labels = []
    all_scores = []
    start_cpu_time = time.process_time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, -score)
        all_labels.append(y)
        all_scores.append(-score)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    # --- 自定义范围切片 ---
    eval_start, eval_end = config.get('auc_range', [1, len(all_labels)])
    sliced_labels = all_labels[eval_start-1 : eval_end]
    sliced_scores = all_scores[eval_start-1 : eval_end]

    roc_auc = roc_auc_score(sliced_labels, sliced_scores)
    threshold = np.median(sliced_scores)
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
              f'sampling_points: {config["argument"]["sampling_points"]}\n'
              f'decay: {config["argument"]["decay"]}\n'
              f'num_components: {config["argument"]["num_components"]}\n'
              f'num_hash_fns: {config["argument"]["num_hash_fns"]}\n',
              end='\n\n',
              file=f)
        
def run_rshash_detector(data, label, config_args):
    """
    封装 RSHash 的运行过程，确保特征边界维度正确
    """
    from pysad.models import RSHash
    from pysad.utils import ArrayStreamer
    import numpy as np

    # --- 关键修正：确保 data 是纯数值 Numpy 数组 ---
    if hasattr(data, 'values'):
        data = data.values
    data = np.asarray(data, dtype=np.float64)
    # ------------------------------------------

    # 1. 显式计算每一列（特征）的最小值和最大值
    f_mins = np.min(data, axis=0)
    f_maxes = np.max(data, axis=0)

    # 2. 初始化模型
    model = RSHash(
        feature_mins=f_mins,
        feature_maxes=f_maxes,
        **config_args
    )
    
    # 3. 设置迭代器
    iterator = ArrayStreamer(shuffle=False)
    all_scores = []
    
    # 4. 模拟流式处理
    for x, _ in iterator.iter(data, label):
        # 修正：使用 np.ravel() 代替 .flatten()，它对 list 和 array 都更友好
        # 或者直接传给模型，因为我们已经确保了 data 是 numpy 格式
        score = model.fit_score_partial(np.ravel(x))
        # 重要：取负号，使“高分”对应“异常”
        all_scores.append(-score)
        
    return np.array(all_scores).flatten()