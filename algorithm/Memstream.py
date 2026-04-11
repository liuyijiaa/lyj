import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
import time
import hdf5storage
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def auc_roc_pr(label, prob):
    """ 计算ROC曲线的AUC值 """
    ROC_area = metrics.roc_auc_score(label, prob)
    '''计算PR曲线的AUC值'''
    precision, recall, _thresholds = metrics.precision_recall_curve(
        label, prob, pos_label=1)
    PR_area = metrics.auc(recall, precision)
    ap = metrics.average_precision_score(label, prob, pos_label=1)
    if ROC_area < 0.5:
        ROC_area = 1 - ROC_area
    if PR_area < 0.5:
        PR_area = 1 - PR_area
    if ap < 0.5:
        ap = 1 - ap
    return round(ROC_area, 4)  # , round(PR_area, 2), round(ap, 2)


def get_data(data_path, data_name, beta, memlen):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=data_name)
    parser.add_argument('--beta', type=float, default=beta)
    parser.add_argument("--dev", help="device", default="cuda:0")
    parser.add_argument("--epochs", type=int,
                        help="number of epochs for ae", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
    parser.add_argument("--memlen", type=int,
                        help="size of memory", default=memlen)
    parser.add_argument("--seed", type=int, help="random seed", default=2)
    args = parser.parse_args(args=[])

    torch.manual_seed(args.seed)
    df = hdf5storage.loadmat(os.path.join(data_path, 'mat', data_name+'.mat'))
    numeric = torch.FloatTensor(df['Y'])
    labels = (df['L']).astype(float).reshape(-1)
    device = torch.device(args.dev)
    return numeric, labels, args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim*2
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(device)
        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0

    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(
                new + 0.001*torch.randn_like(new).to(device)))
            loss = self.loss_fn(output, new)
            loss.backward()
            self.optimizer.step()

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder(new)
        self.memory.requires_grad = False
        self.mem_data = x

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)
        loss_values = torch.norm(
            self.memory - encoder_output, dim=1, p=1).min()
        self.update_memory(loss_values, encoder_output, x)
        return loss_values


def main(config):
    numeric, labels, args = get_data(data_path=config['input path'],
                                     data_name=config['input file'],
                                     beta=config['argument']['beta'],
                                     memlen=config['argument']['memlen'])
    torch.manual_seed(args.seed)
    N = args.memlen
    params = {
        'beta': args.beta, 'memory_len': N, 'batch_size': 1, 'lr': args.lr
    }
    model = MemStream(numeric[0].shape[0], params).to(device)

    batch_size = params['batch_size']
    # print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
    data_loader = DataLoader(numeric, batch_size=batch_size)
    init_data = numeric[:N].to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))
    err = []
    start_cpu_time = time.process_time()
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    total_cpu_time = time.process_time() - start_cpu_time
    scores = np.array([i.cpu() for i in err])
    # --- 自定义范围切片 ---
    eval_start, eval_end = config.get('auc_range', [1, len(labels)])
    sliced_labels = labels[eval_start-1 : eval_end]
    sliced_scores = scores[eval_start-1 : eval_end]

    roc = metrics.roc_auc_score(sliced_labels, sliced_scores)
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
              f'ROC: {roc:.4f}\n'
              f'Precision: {precision:.4f}\n'
              f'Recall: {recall:.4f}\n'
              f'F1: {f1:.4f}\n' 
              f'False Positive Rate (FPR): {fpr:.4f}\n'
              f'CPU Time: {total_cpu_time:.4f} seconds\n'
              f'beta: {config["argument"]["beta"]}\n'
              f'memlen: {config["argument"]["memlen"]}\n',
              end='\n\n',
              file=f)
        
def run_memstream_detector(data, labels, config_args):
    import torch
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    import numpy as np
    
    # 修复 point 1: 确保 data 是 numpy 数组，处理 'list' 没有 astype 的问题
    if isinstance(data, list):
        data = np.array(data)
    data = data.astype(np.float32) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 修复 point 2: 构造符合类定义的 params 字典
    mem_len = config_args.get('memlen', 100)
    params = {
        'beta': config_args.get('beta', 0.1),
        'memory_len': mem_len,
        'lr': config_args.get('lr', 1e-2),
        'batch_size': 1
    }
    
    # 修复 point 3: 正确调用初始化 (in_dim, params)
    input_dim = data.shape[1]
    model = MemStream(input_dim, params).to(device)
    
    # 预训练逻辑 (必须包含，否则会报 'mean' 属性缺失)
    data_tensor = torch.from_numpy(data).to(device)
    init_data = data_tensor[:mem_len]
    model.mem_data = init_data
    
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data), epochs=config_args.get('epochs', 50))
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data))
    
    all_scores = []
    data_loader = DataLoader(data_tensor, batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for x in data_loader:
            output = model(x)
            # 确保将 Tensor 标量转为 Python 浮点数
            all_scores.append(output.cpu().item())
            
    return np.array(all_scores)