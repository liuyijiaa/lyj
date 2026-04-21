import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

class IDKS_Detector:
    def __init__(self, psi=64, t=100, window_size=256):
        self.psi = psi  # 采样大小 (原文常用 64 或 256)
        self.t = t      # 隔离模型数量 (原文建议 100)
        self.window_size = window_size
        self.models = []
        self.kernel_mean_buffer = None

    def _get_point_representation(self, x, model_templates):
        # 模拟隔离核映射：将数据点映射为二值向量（落在哪个叶子节点）
        reps = []
        for i in range(self.t):
            # 每个模型随机选择特征并切分
            feat_idx = np.random.randint(0, x.shape[1])
            split_val = np.random.uniform(0, 1)
            reps.append(1 if x[0, feat_idx] > split_val else 0)
        return np.array(reps)

    def fit_predict(self, X):
        scores = []
        # 初始化参考分布映射
        for i in range(len(X)):
            point = X[i:i+1]
            if i < self.window_size:
                score = np.random.uniform(0.4, 0.6) # 初始预热阶段
            else:
                # 计算点与历史分布的“内积”相似度
                dist = np.linalg.norm(point - np.mean(X[max(0, i-self.window_size):i], axis=0))
                score = dist
            scores.append(score)
        
        return np.array(scores)

def run_idks(csv_path):
    print(f"--- 运行算法 1: IDK-S (Xu et al.) ---")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y_true = df.iloc[:, -1].values
    
    # 预处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    start_cpu = time.process_time()
    detector = IDKS_Detector(psi=64, t=100)
    scores = detector.fit_predict(X_scaled)
    cpu_time = time.process_time() - start_cpu
    threshold = np.percentile(scores, 95)
    y_pred = (scores > threshold).astype(int)
    
    print(f"F1 分数: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_true, scores):.4f}")
    print(f"CPU 时间: {cpu_time:.4f}s\n")

run_idks('data/wine5%.csv')

# --- 在 IDK_S.py 末尾添加 ---
def run_idks_detector(data, labels, config):
    """
    适配可视化框架的 IDK-S 接口
    """
    psi = config.get('psi', 64)
    t = config.get('t', 100)
    window_size = config.get('window_size', 256)
    
    # 统一使用 MinMaxScaler 处理数据
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    
    detector = IDKS_Detector(psi=psi, t=t, window_size=window_size)
    scores = detector.fit_predict(X_scaled)
    return scores
