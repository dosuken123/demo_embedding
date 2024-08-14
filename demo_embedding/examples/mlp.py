import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('demo_embedding/examples/clean.csv')
print(df.head())

X = df.drop('gdp', axis=1).values
y = df['gdp'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class factbook_data:
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)
    

dataset = factbook_data(X_train, y_train, scale_data=False)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

mlp = MLP()
loss_function = nn.L1Loss()
optimizer = torch.optim.Adagrad(mlp.parameters(), lr=1e-4)

for epoch in range(5):
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch} | step: {i:2d} | loss: {loss}")
        
test_data = torch.from_numpy(X_test).float()
test_targets = torch.from_numpy(y_test).float()

mlp.eval()

with torch.no_grad():
    outputs = mlp(test_data)
    predicted_labels = outputs.squeeze().tolist()

predicted_labels = np.array(predicted_labels)
test_targets = np.array(test_targets)

mse = mean_squared_error(test_targets, predicted_labels)
r2 = r2_score(test_targets, predicted_labels)
print(f"mse: {mse}")
print(f"r2: {r2}")