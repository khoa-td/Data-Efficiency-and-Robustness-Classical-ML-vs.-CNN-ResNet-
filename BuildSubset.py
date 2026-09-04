import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import numpy as np
import torchvision.transforms as tt

np.random.seed(42)

train_dataset = CIFAR10(root = 'data/', download= False, train= True, transform= tt.ToTensor())
test_dataset = CIFAR10(root = 'data/', download= False, train= False, transform= tt.ToTensor())

train_targets = np.array(train_dataset.targets)
train_indices = []

test_targets = np.array(test_dataset.targets)
test_indices = []


for label in range(10):
    train_idx = np.where(train_targets == label)[0]
    test_idx = np.where(test_targets == label)[0]

    train_sampled = np.random.choice(train_idx, 1000, replace= False)
    test_sampled = np.random.choice(test_idx, 200, replace= False)

    train_indices.extend(train_sampled)
    test_indices.extend(test_sampled)


X_train = train_dataset.data[train_indices]
y_train = np.array(train_dataset.targets)[train_indices]

X_test = test_dataset.data[test_indices]
y_test = np.array(test_dataset.targets)[test_indices]

np.savez_compressed(
    "cifar10_subset.npz",
    X_test = X_test,
    y_test = y_test,
    X_train = X_train,
    y_train = y_train
)

#test
print(np.bincount(y_train))
print(np.bincount(y_test))


