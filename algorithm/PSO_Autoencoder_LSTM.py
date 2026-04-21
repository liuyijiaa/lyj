import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso

def algorithm_2_pso_ae_lstm(df):
    start_cpu = time.process_time()
    
    # 1. 准备数据
    X = df.iloc[:, :-1].values
    y_true = df.iloc[:, -1].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    time_steps = 5
    X_seq = []
    for i in range(len(X_scaled) - time_steps + 1):
        X_seq.append(X_scaled[i:i+time_steps])
    X_seq = np.array(X_seq)
    y_true_seq = y_true[time_steps-1:]


    input_dim = X_seq.shape[2]
    
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=(time_steps, input_dim), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(time_steps),
        # Decoder
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_dim, activation='sigmoid'))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_seq, X_seq, epochs=2, batch_size=128, verbose=0) # 演示设为2次提高速度

    predictions = model.predict(X_seq)
    mse = np.mean(np.power(X_seq - predictions, 2), axis=(1, 2))

    def fitness_function(threshold):
        y_pred = (mse > threshold).astype(int)
        return -f1_score(y_true_seq, y_pred)

    lb = [np.min(mse)]
    ub = [np.max(mse)]

    best_threshold, _ = pso(fitness_function, lb, ub, swarmsize=10, maxiter=5)
    
    y_final_pred = (mse > best_threshold).astype(int)
    cpu_time = time.process_time() - start_cpu
    
    # 指标计算
    f1 = f1_score(y_true_seq, y_final_pred)
    auc = roc_auc_score(y_true_seq, mse)
    
    print("\n--- 算法 2: PSO-Autoencoder-LSTM ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"CPU Time: {cpu_time:.4f}s")
    print(f"Optimal Threshold (via PSO): {best_threshold[0]:.6f}")

df = pd.read_csv('data/cardio.csv')
algorithm_2_pso_ae_lstm(df)

def run_pso_ae_lstm_detector(data, labels, config={}):
    X_scaled = data
    X_seq = np.expand_dims(X_scaled, axis=1) 
    
    input_dim = X_seq.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(2, input_shape=(1, input_dim), return_sequences=False),
        tf.keras.layers.RepeatVector(1),
        tf.keras.layers.LSTM(2, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, X_seq, epochs=config.get('epochs', 2), batch_size=4096, verbose=0)
    
    preds = model.predict(X_seq)
    scores = np.mean(np.power(X_seq - preds, 2), axis=(1, 2))
    return scores
