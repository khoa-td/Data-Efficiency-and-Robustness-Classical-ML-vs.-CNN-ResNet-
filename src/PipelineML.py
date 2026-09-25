import numpy as np
from skimage.feature import hog
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time




class PipelineML():
    def __init__(self, model_name, seed, n_components):
        Model = {
            'svm': LinearSVC(random_state=seed, max_iter=2000, dual=False),
            'RF': RandomForestClassifier(random_state=seed, n_estimators=100),
            'logistic': LogisticRegression(random_state=seed, max_iter=1000)
        }
        self.name = model_name
        self.model = Model[model_name]
        self.pca = PCA(n_components= n_components)
        self.scaler = StandardScaler()


    def train(self, X_train, y_train):

        X_scaled = self.scaler.fit_transform(X_train)
        
        X_train_pca = self.pca.fit_transform(X_scaled)
        
        self.model.fit(X_train_pca, y_train)

    def evaluate(self, X, y):


        X_scl = self.scaler.transform(X)

        X_pca = self.pca.transform(X_scl)

        y_pred = self.model.predict(X_pca)

        return np.mean(y_pred == y)



if __name__ == '__main__':
    start_time = time.time()
    data = np.load("cifar10_subset.npz")

    X_train = data['X_train']
    y_train = data['y_train']

    X_test = data['X_test']
    y_test = data['y_test']

    def ExactHog(X):
        X_hog = []
        for pic in X:
            feature = hog(pic, orientations = 9, pixels_per_cell = (4, 4), cells_per_block = (2, 2), block_norm = "L2-Hys", channel_axis = -1)
            X_hog.append(feature)

        return np.array(X_hog)

    X_train = ExactHog(X_train)
    

    
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


    def add_noise(images, sigma):
        noise = np.random.normal(scale= sigma, size= images.shape)
        noisy_images = np.clip(images + noise, 0.0, 255.0)

        return noisy_images

    X_test = add_noise(X_test, 0.15 * 255)
    X_test = ExactHog(X_test)


    r_list = [0.05, 1.0]
    n_list = [20, 50, 100]
    models = ["RF", "logistic", "svm"]

    res = []


    for r in r_list:
        X_sub, y_sub = percentage_data(r, X_train, y_train, 42)

        for n_comp in n_list:
            for name in models:

                term = PipelineML(name, 42, n_comp)
                term.train(X_sub, y_sub)

                acc = term.evaluate(X_test, y_test)

                res.append({
                    "Ratio" : r,
                    "n_components" : n_comp,
                    "Model" : name,
                    "Accuracy %" : round(acc * 100, 2) 
                })

    df_res = pd.DataFrame(res)
    print(df_res.to_string(index=False))

    print("Run time:", time.time() - start_time)

