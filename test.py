import cv2
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import torch
from collections import deque
from random import random
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn


w = 0.7
b = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step)
Y = X*w + b

split = int(len(X) * 0.8)

X_train, y_train, X_test, y_test = X[:split], Y[:split], X[split:], Y[split:]

def plot_show(train_data=X_train,
              train_label=y_train,
              test_data=X_test,
              test_label=y_test,
              predictions=None):
    
    plt.figure(figsize=(8, 5))
    plt.scatter(train_data, train_label, c='y', s=4, label="Train Data")
    plt.scatter(test_data, test_label, c='b', s=4, label="Test Data")
    if predictions is not None:
        plt.scatter(predictions, test_label, c='r', s=4, label="Predictions")
    plt.legend()
    plt.show()
    


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1,
                                         requires_grad=True,
                                         dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1,
                                         requires_grad=True,
                                         dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x + self.b
    

torch.manual_seed(42)
model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
# print(model_0.state_dict())

# with torch.inference_mode():
#     pred = model_0(X_test)
# plot_show(predictions=pred)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.01)

torch.manual_seed(42)

epochs = 200
train_losses = []
test_losses = []

for epoch in range(epochs):
    model_0.train()
    
    y_pred = model_0(X_train)
    
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        train_losses.append(loss)
        test_losses.append(test_loss)
        print(f'epoch : {epoch}, train loss :{loss}, test loss :{test_loss}')
        print(model_0.state_dict())

with torch.inference_mode():
    pred = model_0(X_test)
plot_show(predictions=pred)