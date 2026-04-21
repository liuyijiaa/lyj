import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class TS_Dataset(Dataset):
    def __init__(self, data, labels, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # 实时切片窗口
        window = self.data[idx : idx + self.window_size]
        label = self.labels[idx + self.window_size]
        return window, label

class LSTM_CNN_Model(nn.Module):
    def __init__(self, n_features):
        super(LSTM_CNN_Model, self).__init__()
        # CNN: [Batch, Features, Window]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, n_features)

    def forward(self, x):
        # x shape: [Batch, Window, Features] -> [Batch, Features, Window]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # x shape: [Batch, 64, Window/2] -> [Batch, Window/2, 64]
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :]) # 取最后一个时间步预测

def run_lstm_cnn(csv_path):
    print(f"--- 运行算法 2: LSTM-CNN (Duraj et al.) ---")
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    

    window_size = 30
    batch_size = 512 # 小批量处理，防止 OOM
    
    dataset = TS_Dataset(X_scaled, y_raw, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM_CNN_Model(X_raw.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start_cpu = time.process_time()
    
    # 训练阶段
    model.train()
    for epoch in range(5): # UNSW 数据集大，5轮即可见效
        for batch_x, _ in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            # 预测下一个时间步的任务
            loss = criterion(output, batch_x[:, -1, :]) 
            loss.backward()
            optimizer.step()

    # 推理阶段 (分批次计算得分，防止内存溢出)
    model.eval()
    scores = []
    y_test = []
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            preds = model(batch_x)
            batch_scores = torch.mean((preds - batch_x[:, -1, :])**2, dim=1)
            scores.extend(batch_scores.numpy())
            y_test.extend(batch_y.numpy())
    
    cpu_time = time.process_time() - start_cpu
    
    # 评估指标
    scores = np.array(scores)
    threshold = np.percentile(scores, 95)
    y_pred = (scores > threshold).astype(int)
    
    print(f"F1 分数: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_test, scores):.4f}")
    print(f"CPU 时间: {cpu_time:.4f}s")
    print("-" * 40)

run_lstm_cnn('data/cardio.csv')


def run_lstm_cnn_detector(data, labels, config):
    """
    适配可视化框架的 LSTM-CNN 接口
    """
    window_size = config.get('window_size', 30)
    epochs = config.get('epochs', 5)
    batch_size = config.get('batch_size', 512)
    
    # 标准化预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    dataset = TS_Dataset(X_scaled, labels, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM_CNN_Model(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 简易训练流程
    model.train()
    for _ in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x[:, -1, :])
            loss.backward()
            optimizer.step()

    # 推理阶段
    model.eval()
    scores = []
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_x, _ in eval_loader:
            preds = model(batch_x)
            batch_scores = torch.mean((preds - batch_x[:, -1, :])**2, dim=1)
            scores.extend(batch_scores.numpy())
    

    full_scores = np.zeros(len(data))
    full_scores[window_size:] = np.array(scores)
    return full_scores
