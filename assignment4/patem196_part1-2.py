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
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(batch_size):
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])

    # Download and load the training dataset
    train_dataset = datasets.FashionMNIST(
        root='./data',  # Path to save/download the dataset
        train=True,     # Load the training set
        transform=transform,  # Apply the defined transformations
        download=True   # Download the dataset if not already downloaded
    )

    # Download and load the test dataset
    test_data = datasets.FashionMNIST(
        root='./data',
        train=False,    # Load the test set
        transform=transform,
        download=True
    )

    # Create DataLoaders for batch processing
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Example: Displaying some dataset info
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of Validation samples: {len(val_data)}")
    print(f"Number of testing samples: {len(test_data)}")
    return train_loader, val_loader, test_loader


class FMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)  # Conv_10
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3)  # Conv_5
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)  # Conv_16
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling with filter size 2
        
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Convolution + ReLU + MaxPooling layers in the specified order
        x = F.relu(self.conv1(x))   # Conv_10 -> ReLU
        x = self.pool(x)            # Pooling

        x = F.relu(self.conv2(x))   # Conv_5 -> ReLU
        x = F.relu(self.conv3(x))   # Conv_16 -> ReLU
        x = self.pool(x)            # Pooling again

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 4 * 4)  # Flatten the output of the last pooling layer
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))     # FC -> ReLU
        x = F.relu(self.fc2(x))     # FC -> ReLU
        x = self.fc3(x)      
        
        return x
def train(model, train_loader, val_loader, epochs, optimizer, criterion, patience=3, tolerance=1e-4):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=patience, threshold=tolerance)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping if the validation loss doesn't improve
        if val_loss < best_val_loss - tolerance:
            best_val_loss = val_loss

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses, epoch + 1

def run_train(batch_size, learning_rate, epochs):

    train_loader, val_loader, test_loader = load_data(batch_size)
    model = FMNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses, val_losses, epochs_used = train(model, train_loader, val_loader, epochs, optimizer, criterion)

    plt.figure()
    plt.plot(range(epochs_used), train_losses, label='Train Loss')
    plt.plot(range(epochs_used), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{batch_size}_{learning_rate}.png')
    plt.close()
    
    return model, test_loader

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def batch_size_experiment(batch_sizes, learning_rate, epochs):

    # To store the results
    batch_size_accuracy = []

    for idx, batch_size in enumerate(batch_sizes):
        print(f"Training with batch size: {batch_size}, Learning Rate: {learning_rate[idx]}")
        model, test_loader = run_train(batch_size, learning_rate[idx], epochs)
        model_accuracy = evaluate(model, test_loader)
        batch_size_accuracy.append((batch_size, model_accuracy))

    # Extract batch sizes and accuracies for plotting
    batch_sizes = [x[0] for x in batch_size_accuracy]
    accuracies = [x[1] for x in batch_size_accuracy]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, accuracies)
    plt.title('Batch Size vs Accuracy, Epochs = 45')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy (%)')
    plt.grid()
    plt.show()
    return batch_size_accuracy
# Run evaluation
def part1():
    batch_size_experiment([16], [0.005], 45)

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
def part2():
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

part1()
part2()