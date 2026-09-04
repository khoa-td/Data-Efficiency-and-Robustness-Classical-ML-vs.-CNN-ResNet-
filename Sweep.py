import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from skimage.feature import hog



from PipelineResNet import PipelineCNN, SmallResNet
from PipelineML import PipelineML


def get_indices(r, y, seed):
    count = int(r * len(y)) // 10
    X_idx = []

    np.random.seed(seed)
    for i in range(10):
        idx = np.where(y == i)[0]
        sample = np.random.choice(idx, count, replace= False)

        X_idx.extend(sample)

    return X_idx

def ExactHog(X):
    X_hog = []
    for pic in X:
        feature = hog(pic, orientations = 9, pixels_per_cell = (4, 4), cells_per_block = (2, 2), block_norm = "L2-Hys", channel_axis = -1)
        X_hog.append(feature)

    return np.array(X_hog)

def add_noise(img, sigma):
    noise = np.random.normal(scale= sigma, size= img.shape)

    return np.clip(img + noise, 0.0, 255.0)



if __name__ == '__main__':
    data = np.load('cifar10_subset.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    X_train_hog = ExactHog(X_train)

    sigmas = [0, 12.75, 38.25, 76.5]
    test_data = {}

    for sigma in sigmas:
        X_test_noise_img = add_noise(X_test, sigma)
        X_test_noise_hog = ExactHog(X_test_noise_img)

        test_data[sigma] = {'CNN' : X_test_noise_img, 'ML': X_test_noise_hog}

    res = []
    file_name = "Schema Log.xlsx"

    percentages= [0.05, 0.1, 0.25, 0.5, 1.0]
    seeds = [42, 43, 44]

    for r in percentages:
        for s in seeds:
            X_idx = get_indices(r, y_train, s)
            y_sub = y_train[X_idx]

            X_sub_CNN = X_train[X_idx]
            X_sub_Ml = X_train_hog[X_idx]

            for model_name in ['svm', 'logistic', 'RF']:
                start_time = time.time()
                model_ml = PipelineML(model_name, s, n_components= 50)
                model_ml.train(X_sub_Ml, y_sub)

                train_time = time.time() - start_time

                for sigma in sigmas:
                    acc = model_ml.evaluate(test_data[sigma]['ML'], y_test)

                    res.append({
                        "Pipeline": "Machine Learning",
                        "Model Name": model_name,
                        "Percentage Data": r,
                        "Seed": s,
                        "Noise Sigma": sigma,
                        "Accuracy": round(acc * 100, 2),
                        "Train Time": round(train_time, 2)
                    })


            start_time = time.time()
            term_cnn = PipelineCNN(epochs= 10, batch_size= 64, optimizer_cls= torch.optim.Adam, loss_fn= nn.CrossEntropyLoss(), lr= 0.001, seed= s)
            model_cnn = SmallResNet()

            model_cnn = term_cnn.train(model_cnn, X_sub_CNN, y_sub)
            train_time = time.time() - start_time

            for sigma in sigmas:
                X_test_tensor = torch.tensor(test_data[sigma]['CNN'], dtype= torch.float32).permute(0, 3, 1, 2) / 255.0
                y_test_tensor = torch.tensor(y_test, dtype= torch.long)

                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, 64, shuffle= False)

                _, acc = term_cnn.evaluate(model_cnn, test_loader)

                res.append({
                    "Pipeline": "Deep Learning",
                    "Model Name": "SmallResNet",
                    "Percentage Data": r,
                    "Seed": s,
                    "Noise Sigma": sigma,
                    "Accuracy": round(acc * 100, 2),
                    "Train Time": round(train_time, 2)
                })

        df_res = pd.DataFrame(res)
        df_res.to_excel(file_name, index= False, sheet_name="Data")




