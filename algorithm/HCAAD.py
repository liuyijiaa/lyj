import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.fft import fft
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class HCAAD_Model(nn.Module):
    def __init__(self, n_features):
        super(HCAAD_Model, self).__init__()
        # 增加隐藏层以提高容量，符合原文 Hierarchical 结构
        self.encoder = nn.Linear(n_features, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.decoder = nn.Linear(64, n_features)

    def forward(self, x):
        # x shape: [batch, features] -> [batch, 1, features] 为适配注意力层
        x_enc = torch.relu(self.encoder(x)).unsqueeze(1)
        attn_out, _ = self.attn(x_enc, x_enc, x_enc)
        out = self.decoder(attn_out.squeeze(1))
        return out

def run_hcaad(csv_path):
    print(f"--- 运行算法 2: HCAAD (内存优化版) ---")
    
    # 1. 数据加载与清洗 (针对 UNSW_NB15)
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_true = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. 模拟 HCAAD 的频域组件感知
    X_fft = np.abs(fft(X_scaled, axis=0))
    X_combined = torch.tensor(X_scaled + 0.1 * X_fft)
    
    # 3. 准备分批次数据 (Batch Size 设为 512 或 1024 避免内存崩溃)
    dataset = TensorDataset(X_combined)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = HCAAD_Model(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    start_cpu = time.process_time()
    
    # 4. 训练阶段
    model.train()
    for epoch in range(10): # 针对大数据集，减少 epoch 提高速度
        total_loss = 0
        for batch in loader:
            batch_data = batch[0]
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

    # 5. 推理阶段 (按批次计算得分，防止预测时内存溢出)
    model.eval()
    scores = []
    eval_loader = DataLoader(TensorDataset(X_combined), batch_size=1024, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            batch_data = batch[0]
            preds = model(batch_data)
            batch_scores = torch.mean((preds - batch_data)**2, dim=1)
            scores.extend(batch_scores.numpy())
    
    scores = np.array(scores)
    cpu_time = time.process_time() - start_cpu
    
    # 6. 评估指标
    # 使用 95 分位数作为异常阈值
    threshold = np.percentile(scores, 95)
    y_pred = (scores > threshold).astype(int)
    
    print(f"\n结果统计:")
    print(f"F1 分数: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_true, scores):.4f}")
    print(f"CPU 时间: {cpu_time:.4f}s")
    print("-" * 40)

run_hcaad('data/UNSW_NB15_training-set.csv')

def run_hcaad_detector(data, labels, config={}):
    # 模拟 run_hcaad 内部逻辑，但适配可视化脚本的 data 输入
    n_features = data.shape[1]
    model = HCAAD_Model(n_features)
    
    # 转换数据
    X_tensor = torch.tensor(data.astype(np.float32))
    loader = DataLoader(TensorDataset(X_tensor), batch_size=128, shuffle=True)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(config.get('epochs', 5)): # 默认跑5轮提高速度
        for batch in loader:
            batch_data = batch[0]
            optimizer.zero_grad()
            loss = criterion(model(batch_data), batch_data)
            loss.backward()
            optimizer.step()
            
    # 计算得分
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        scores = torch.mean((preds - X_tensor)**2, dim=1).numpy()
    return scores
