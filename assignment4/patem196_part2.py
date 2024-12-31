# Author : Meet patel
# Date: December 1st

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
def print_bias_analysis(Y_true, Y_pred, group, fpr, tpr):
    tn, fp, fn, tp = confusion_matrix(Y_true,Y_pred).ravel()
    print(f"\n\nFor {group}:")
    print(f"Total cases: {len(Y_true)}, Chance of recidivism: {((tp + fp)/len(Y_true)):.2%}, False positive rate: {(fpr):.2%}, True positive rate: {(tpr):.2%}")
    
def equalized_odds(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm.ravel()
    tpr = tp / (tp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    return tpr, fpr

def find_most_disadvantaged_group(groups_dict):
    ds_scores = {group: 0 for group in groups_dict.keys()}
    results = dict()
    for x in groups_dict.keys():
        for y in groups_dict.keys():
            if x != y:
                tpr_diff = abs(groups_dict[x][0] - groups_dict[y][0])
                fpr_diff = abs(groups_dict[x][1] - groups_dict[y][1])
                results[(x,y)] = (tpr_diff,fpr_diff)
                ds_scores[x] += tpr_diff + fpr_diff
    
    most_disadvantaged_group = max(ds_scores, key=ds_scores.get)

    return most_disadvantaged_group


def balance_data (df, most_disadvantaged_group):
    disadvantaged_data = df[df['race'] == most_disadvantaged_group]

    class_0 = disadvantaged_data[disadvantaged_data['score'] == 0]
    class_1 = disadvantaged_data[disadvantaged_data['score'] == 1]

    min_count = min(len(class_0), len(class_1))
    balanced_subgroup = pd.concat([
        class_0.sample(n=min_count, random_state=42),
        class_1.sample(n=min_count, random_state=42)
    ])

    other_data = df[df['race'] != most_disadvantaged_group]
    balanced_dataset = pd.concat([other_data, balanced_subgroup]).reset_index(drop=True)

    return balanced_dataset

epochs = 100
learning_rate = 0.01

file_path = "compas-scores.csv"
df = pd.read_csv(file_path)

df['score'] = df['score_text'].map({'Low': 0, 'Medium': 0, 'High': 1})
relevant_columns = ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'score', 'sex', 'is_recid', 'r_charge_degree', 'decile_score']
df_cleaned = df[relevant_columns].dropna()

X = df_cleaned.drop(columns=['score'])
Y = df_cleaned['score']

categorical_features = ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc', 'sex', 'is_recid', 'r_charge_degree']
continuous_features = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'decile_score']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_categorical = encoder.fit_transform(X[categorical_features])
X_continuous = scaler.fit_transform(X[continuous_features].values.reshape(-1, len(continuous_features)))    

X_processed = np.hstack([X_categorical, X_continuous])
X_processed_df = pd.DataFrame(X_processed, index=df_cleaned.index)
X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=42)

model = LogisticRegressionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    y_pred_train = (model(X_train_tensor) >= 0.5).float()
    y_pred_test = (model(X_test_tensor) >= 0.5).float()

train_accuracy = accuracy_score(Y_train, y_pred_train.numpy())
test_accuracy = accuracy_score(Y_test, y_pred_test.numpy())
print(f"Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")      

races = dict()
grouped = df_cleaned.groupby('race')
for group, data in grouped:
    indices = data.index
    
    y_true = Y.loc[indices] 
    X_group = X_processed_df.loc[indices]
    
    model_input = torch.tensor(X_group.values, dtype=torch.float32)
    y_pred = (model(model_input).detach().numpy() >= 0.5).astype(int).flatten()
    
    # Calculate Equalized Odds metrics
    tpr, fpr = equalized_odds(y_true, y_pred)
    races[group] = (tpr,fpr)
    print_bias_analysis(y_true, y_pred, group, fpr, tpr)        

    
most_disadvantaged_group = find_most_disadvantaged_group(races)


df_balanced = balance_data(df_cleaned, most_disadvantaged_group)

X_balanced = df_balanced.drop(columns=['score'])
Y_balanced = df_balanced['score']

X_categorical_bal = encoder.transform(X_balanced[categorical_features])
X_continuous_bal = scaler.transform(X_balanced[continuous_features])
X_balanced_processed = np.hstack([X_categorical_bal, X_continuous_bal])

X_categorical = encoder.fit_transform(X[categorical_features])
X_continuous = scaler.fit_transform(X[continuous_features].values.reshape(-1, len(continuous_features)))    

X_processed = np.hstack([X_categorical, X_continuous])
X_processed_df = pd.DataFrame(X_balanced_processed, index=df_balanced.index)
    
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced_processed, Y_balanced, test_size=0.2, random_state=42)

model = LogisticRegressionModel(X_train_bal.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train_bal_tensor = torch.tensor(X_train_bal, dtype=torch.float32)
y_train_bal_tensor = torch.tensor(y_train_bal.values, dtype=torch.float32).view(-1, 1)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_bal_tensor)
    loss = criterion(outputs, y_train_bal_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        
model.eval()
with torch.no_grad():
    # Get predictions for balanced training and test sets
    y_pred_train_bal = (model(X_train_bal_tensor) >= 0.5).float()
    y_pred_test_bal = (model(torch.tensor(X_test_bal, dtype=torch.float32)) >= 0.5).float()
    
    # Calculate accuracies for balanced dataset
    train_accuracy_bal = accuracy_score(y_train_bal, y_pred_train_bal.numpy())
    test_accuracy_bal = accuracy_score(y_test_bal, y_pred_test_bal.numpy())
    print(f"\nBalanced Dataset Metrics:")
    print(f"Balanced Train Accuracy: {train_accuracy_bal:.2f}")
    print(f"Balanced Test Accuracy: {test_accuracy_bal:.2f}\n")
    
    # Get predictions for full balanced dataset
    y_pred_bal = (
        model(torch.tensor(X_balanced_processed, dtype=torch.float32)).detach().numpy() >= 0.5
    ).astype(int).flatten()

# Calculate new Equalized Odds using the balanced dataset indices
races = dict()
print("Equalized Odds for Balanced Dataset:")
for group, data in df_balanced.groupby('race'):
    # Get group data
    group_indices = data.index
    
    # Get predictions and true values for this group
    y_true = Y_balanced.iloc[group_indices]
    y_pred = y_pred_bal[group_indices]
    
    # Calculate Equalized Odds metrics
    tpr, fpr = equalized_odds(y_true, y_pred)
    races[group] = (tpr,fpr)
    print_bias_analysis(y_true, y_pred, group, fpr, tpr)      