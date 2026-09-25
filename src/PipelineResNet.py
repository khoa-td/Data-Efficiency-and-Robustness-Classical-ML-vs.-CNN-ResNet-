import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def conv_2d(ni, nf, stride= 1, ks= 3):
    return nn.Conv2d(ni, nf, stride= stride, kernel_size= ks, bias= False, padding=  ks // 2)

def bn_relu_conv(ni, nf, stride= 1, ks= 3):
    return nn.Sequential(
        nn.BatchNorm2d(ni),
        nn.ReLU(),
        conv_2d(ni, nf, stride= stride, ks= ks)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride= 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = conv_2d(in_channels, out_channels, stride= stride)
        self.conv2 = bn_relu_conv(out_channels, out_channels, stride= 1)

        self.shorcut = lambda x: x
        if in_channels != out_channels or stride != 1:
            self.shorcut = conv_2d(in_channels, out_channels, stride= stride, ks= 1)


    def forward(self, x):
        x = F.relu(self.bn(x), inplace= True)
        r = self.shorcut(x)

        x = self.conv1(x)
        x = self.conv2(x) * 0.2

        return x + r


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.full = nn.Sequential(
            conv_2d(3, 8),
            ResidualBlock(8, 16, 2),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.full(x)


class PipelineCNN():
    def __init__(self, epochs, batch_size, optimizer_cls, loss_fn, lr = 1e-3, device='gpu', seed= 42):
        self.seed = seed
        set_seed(self.seed)

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.opt = optimizer_cls
        self.lr = lr
        self.device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')

    def train_epoch(self, model, dataloader, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = self.loss_fn(y_pred, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, pred = torch.max(y_pred, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, model, dataloader):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                y_pred = model(xb)
                loss = self.loss_fn(y_pred, yb)

                running_loss += loss.item() * xb.size(0)
                _, pred = torch.max(y_pred, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, model, X_train, y_train):
        X_tr = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y_tr = torch.tensor(y_train, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)

        model = model.to(self.device)
        optimizer = self.opt(model.parameters(), lr = self.lr)

        best_loss = float('inf')
        best_model = None

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer)

            if train_loss < best_loss:
                best_loss = train_loss
                best_model = copy.deepcopy(model)

            # print(f"Epoch:{epoch} | Loss:{train_loss} | Accuracy:{train_acc}")

        return best_model if best_model is not None else model

if __name__ == '__main__':
    start_time = time.time()

    data = np.load("cifar10_subset.npz")
    
    X_train = data['X_train']
    y_train = data['y_train']

    X_test = data['X_test']
    y_test = data['y_test']

    def add_noise(images, sigma):
        noise = np.random.normal(scale= sigma, size= images.shape)
        noise_images = np.clip(images + noise, 0.0, 255.0)

        return noise_images

    X_test = add_noise(X_test, 0.15 * 255)

    X = torch.tensor(X_test, dtype= torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor(y_test, dtype= torch.long)


    test_data = TensorDataset(X, y)
    test_loader = DataLoader(test_data, 64, shuffle= False)
    
    def percentage_data(r, X, y, seed):

        count = int(r * len(X)) // 10

        X_idx, y_idx = [], []
        np.random.seed(seed)
        for i in range(10):

            idx = np.where(y == i)[0]

            sample = np.random.choice(idx, count, replace= False)

            X_idx.extend(sample)
            y_idx.extend(sample)
        
        
        X_r = X[X_idx]
        y_r = y[y_idx]

        return X_r, y_r


    X_train, y_train = percentage_data(0.5, X_train, y_train, 42)

    term = PipelineCNN(10, 64, optimizer_cls= torch.optim.Adam, loss_fn= nn.CrossEntropyLoss(), lr = 0.001, device= 'gpu')
    model = SmallResNet()

    model = term.train(model, X_train, y_train)
    test_loss, test_acc = term.evaluate(model, test_loader)

    print("Loss:", test_loss)
    print("Accuracy:", test_acc)

    print("Run time:", time.time() - start_time)
