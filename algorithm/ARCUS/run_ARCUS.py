# 1. 顶部添加警告过滤
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import numpy as np
from algorithm.ARCUS.ARCUS import ARCUS
from algorithm.ARCUS.datasets.data_utils import load_dataset
from algorithm.ARCUS.utils import set_gpu, set_seed
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def main(config):
    parser = argparse.ArgumentParser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser.add_argument('--run_config', '-r', default='')
    parser.add_argument('--model_type', type=str, default=config['argument']['model_type'],
                        choices=["RAPP", "RSRAE", "DAGMM"])
    parser.add_argument('--inf_type', type=str, default=config['argument']['inf_type'],
                        choices=["ADP", "INC"], help='INC: drift-unaware, ADP: drift-aware')
    parser.add_argument('--dataset_name', type=str,
                        default=(config['input path'], config['input file']))
    parser.add_argument('--seed', type=int,
                        default=config['argument']['seed'], help="random seed")
    parser.add_argument(
        '--gpu', type=str, default=config['argument']['gpu'], help="foramt is '0,1,2,3'")
    parser.add_argument('--batch_size', type=int,
                        default=config['argument']['batch_size'])
    parser.add_argument('--min_batch_size', type=int,
                        default=config['argument']['min_batch_size'])
    parser.add_argument('--init_epoch', type=int,
                        default=config['argument']['init_epoch'])
    parser.add_argument('--intm_epoch', type=int,
                        default=config['argument']['intm_epoch'])
    parser.add_argument('--hidden_dim', type=int, default=config['argument']['hidden_dim'],
                        help="The hidden dim size of AE. \
                            Manually chosen or the number of pricipal component explaining at least 70% of variance: \
                            MNIST_AbrRec: 24,  MNIST_GrdRec: 25, F_MNIST_AbrRec: 9, F_MNIST_GrdRec: 15, GAS: 2, RIALTO: 2, INSECTS_Abr: 6, \
                            INSECTS_Incr: 7, INSECTS_IncrGrd: 8, INSECTS_IncrRecr: 7")
    parser.add_argument('--layer_num', type=int,
                        default=config['argument']['layer_num'], help="Num of AE layers")
    parser.add_argument('--RSRAE_hidden_layer_size', type=int, nargs="+", default=[32, 64, 128],
                        help="Suggested by the RSRAE author. The one or two layers of them may be used according to data sets")
    parser.add_argument('--learning_rate', type=float,
                        default=config['argument']['learning_rate'])
    parser.add_argument('--reliability_thred', type=float,
                        default=config['argument']['reliability_thred'], help='Threshold for model pool adaptation')
    parser.add_argument('--similarity_thred', type=float,
                        default=config['argument']['similarity_thred'], help='Threshold for model merging')

    args = parser.parse_args()
    args = set_gpu(args)
    set_seed(args.seed)

    start_cpu_time = time.process_time()
    args, loader = load_dataset(args)
    ARCUS_instance = ARCUS(args)
    returned, auc_hist, anomaly_scores = ARCUS_instance.simulator(loader)
    total_cpu_time = time.process_time() - start_cpu_time
    all_true_labels = []
    for batch in loader:
        if isinstance(batch, tuple) and len(batch) >= 2:
            true_label = batch[1]
        else:
            true_label = batch['label']
    # 修复弃用的.cpu()
        true_label = true_label.numpy().flatten()  # 移除.cpu()
        all_true_labels.extend(true_label.tolist())

    all_true_labels = np.array(all_true_labels)
    all_true_labels = np.where(all_true_labels == -1, 1, all_true_labels)
    all_true_labels = np.clip(all_true_labels, 0, 1)

    anomaly_scores = np.array(anomaly_scores)

    # --- 新增功能：自定义范围切片 ---
    # 根据 config 中的 auc_range 获取起止点，默认为全量
    eval_start, eval_end = config.get('auc_range', [1, len(all_true_labels)])
    
    # 截取指定范围内的数据
    sliced_labels = all_true_labels[eval_start-1 : eval_end]
    sliced_scores = anomaly_scores[eval_start-1 : eval_end]

    # --- 基于切片范围重新计算评价指标 ---
    # 1. 生成预测结果（保留原有的中位数阈值逻辑，但在切片后的数据上应用）
    if len(sliced_scores) > 0:
        threshold = np.median(sliced_scores)
        sliced_predictions = (sliced_scores > threshold).astype(int)
    else:
        sliced_predictions = np.array([])

    # 2. 计算 ROC（增加安全性判断）
    if len(np.unique(sliced_labels)) > 1:
        roc_val = roc_auc_score(sliced_labels, sliced_scores)
    else:
        roc_val = 0.5

    # 3. 计算 Precision, Recall, F1, FPR
    precision = precision_score(sliced_labels, sliced_predictions, zero_division=0)
    recall = recall_score(sliced_labels, sliced_predictions, zero_division=0)
    f1 = f1_score(sliced_labels, sliced_predictions, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(sliced_labels, sliced_predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    except:
        fpr = 0.0

    # --- 输出结果（恢复并保留所有原始字段，仅新增 Evaluation Range） ---
    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'Evaluation Range: {eval_start} to {eval_end}\n' # 新增行
              f'ROC: {roc_val:.4f}\n'
              f'Precision: {precision:.4f}\n'
              f'Recall: {recall:.4f}\n'
              f'F1: {f1:.4f}\n'
              f'False Positive Rate (FPR): {fpr:.4f}\n'
              f'CPU Time: {total_cpu_time:.4f} seconds\n'
              f'model_type: {config["argument"]["model_type"]}\n'
              f'inf_type: {config["argument"]["inf_type"]}\n'
              f'seed: {config["argument"]["seed"]}\n'
              f'gpu: {config["argument"]["gpu"]}\n'
              f'batch_size: {config["argument"]["batch_size"]}\n'
              f'min_batch_size: {config["argument"]["min_batch_size"]}\n'
              f'init_epoch: {config["argument"]["init_epoch"]}\n'
              f'intm_epoch: {config["argument"]["intm_epoch"]}\n'
              f'hidden_dim: {config["argument"]["hidden_dim"]}\n'
              f'layer_num: {config["argument"]["layer_num"]}\n'
              f'learning_rate: {config["argument"]["learning_rate"]}\n' # 确保此行存在
              f'reliability_thred: {config["argument"]["reliability_thred"]}\n' # 确保此行存在
              f'similarity_thred: {config["argument"]["similarity_thred"]}\n', # 确保此行存在
              end='\n\n',
              file=f)
def run_arcus_detector(data, label, config_args):
    import argparse
    import numpy as np
    from algorithm.ARCUS.ARCUS import ARCUS
    
    # 1. 鲁棒的数据转换 (处理可能的表头)
    data = np.array(data)
    label = np.array(label)
    if data.dtype.kind in 'SUO': # 字符串或对象类型
        try:
            data = np.nan_to_num(data[1:].astype(np.float32))
            label = np.nan_to_num(label[1:].astype(np.int32))
        except:
            pass # 保持原样交给 detector 处理

    # 2. 配置参数
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args.model_type = config_args.get('model_type', 'RAPP')
    args.inf_type = config_args.get('inf_type', 'ADP')
    args.batch_size = config_args.get('batch_size', 256)
    args.min_batch_size = 64
    args.init_epoch = config_args.get('init_epoch', 20)
    args.intm_epoch = config_args.get('intm_epoch', 10)
    args.input_dim = data.shape[1]
    args.layer_num = config_args.get('layer_num', 3)
    args.learning_rate = config_args.get('learning_rate', 1e-4)
    args.reliability_thred = config_args.get('reliability_thred', 0.9)
    args.similarity_thred = config_args.get('similarity_thred', 0.7)
    args.hidden_dim = config_args.get('hidden_dim', 32)
    args.seed = config_args.get('seed', 42)
    args.gpu = '-1' 

    # 3. 构造仿真 Loader 解决 'int' object is not callable
    class FakeTFDataset:
        def __init__(self, d, l, batch_size):
            self.data = d
            self.label = l
            self.batch_size = batch_size
        
        # 将 batch 定义为方法，返回自己
        def batch(self, batch_size):
            self.batch_size = batch_size
            return self
        
        # 模拟 prefetch 等 TF 常用方法，防止后续链式调用报错
        def prefetch(self, buffer_size):
            return self

        def __iter__(self):
            # 这里的 yield 结构必须符合 ARCUS 内部对 batch 的解析逻辑
            # 如果它预期 (x, y) 元组：
            yield (self.data, self.label)

    # 4. 执行运行
    try:
        loader = FakeTFDataset(data, label, args.batch_size)
        detector = ARCUS(args)
        
        # 调用 simulator
        _, _, anomaly_scores = detector.simulator(loader)
        
        scores = np.array(anomaly_scores).flatten()
        
        # 长度对齐
        if len(scores) < len(data):
            pad_width = len(data) - len(scores)
            scores = np.pad(scores, (pad_width, 0), 'edge')
            
        return scores
    except Exception as e:
        print(f"ARCUS 运行失败: {e}")
        return np.zeros(len(data))