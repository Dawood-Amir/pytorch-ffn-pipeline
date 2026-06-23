import numpy as np 
import torch
import config as config

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis=0)
    # Prevent division by zero if variance is zero
    sigma = np.where(sigma == 0, 1e-8, sigma)
    return (X - mu) / sigma


def get_data_loaders(seed):
    
    iris = load_iris()
    X= normalize(iris.data)
    y = iris.target


    #   Extract the 80% training set. Hold onto 20% for temp data.
    X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    X_val_np , X_test_np , y_val_np, y_test_np = train_test_split(
        X_temp_np , y_temp_np , test_size=0.5 , random_state=seed)

    #  Convert NumPy matrices into Tensor structures 
    # using Zero-Copy references     
    X_train_tensor = torch.from_numpy(X_train_np).float()
    y_train_tensor = torch.from_numpy(y_train_np).long()

    #val 
    X_val_tensor = torch.from_numpy(X_val_np).float()
    y_val_tensor= torch.from_numpy(y_val_np).long()

    X_test_tensor =  torch.from_numpy(X_test_np).float()
    y_test_tensor = torch.from_numpy(y_test_np)


    # Pack tensors into abstract Stream datasets
    trin_dataset = TensorDataset(X_train_tensor , y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor , y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor , y_test_tensor)

    #  Instanciate streaming data loops
    train_loader= DataLoader(trin_dataset, batch_size=config.BATCH_SIZE ,shuffle=True)
    val_loader = DataLoader(val_dataset , batch_size=config.BATCH_SIZE,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader , test_loader