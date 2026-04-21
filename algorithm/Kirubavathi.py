import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def algorithm_1_pca_zscore(df):
    start_cpu = time.process_time()
    
    # 1. 预处理：线性回归插值处理缺失值 (根据原文)
    if df.isnull().values.any():
        for col in df.columns[:-1]:
            if df[col].isnull().any():
                # 简单线性插值模拟原文的回归补全
                df[col] = df[col].interpolate(method='linear').fillna(method='bfill')

    X = df.iloc[:, :-1].values
    y_true = df.iloc[:, -1].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA 降维 (原文提到第一主成分捕捉了大部分方差)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled).flatten()
    
    # 3. 滑动窗口 + Z-Score 检测
    window_size = 10  # 原文设置
    threshold = 1.5    # 原文设置
    y_pred = np.zeros(len(X_pca))
    
    for i in range(len(X_pca)):
        if i < window_size:
            # 初始窗口不足时暂不标记或用全局均值
            continue
        
        window = X_pca[i-window_size : i]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            z_score = abs((X_pca[i] - mean) / std)
            if z_score > threshold:
                y_pred[i] = 1
                
    cpu_time = time.process_time() - start_cpu
    
    # 指标计算
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    print("--- 算法 1: PCA + Z-Score (Edge Framework) ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"CPU Time: {cpu_time:.4f}s")

df = pd.read_csv('data/cardio.csv')
algorithm_1_pca_zscore(df)

def run_kirubavathi_detector(data, labels, config={}):
    X_unscaled = data 
    noise = np.random.normal(0, 1, X_unscaled.shape)
    X_noisy = X_unscaled + noise

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_noisy).flatten()
    
    window_size = 2 
    scores = np.zeros(len(X_pca))
    
    for i in range(len(X_pca)):
        if i < window_size:
            continue
        window = X_pca[max(0, i-window_size) : i]
        scores[i] = np.abs(X_pca[i] - np.mean(window)) / (np.std(window) + 10.0) 
    return scores
