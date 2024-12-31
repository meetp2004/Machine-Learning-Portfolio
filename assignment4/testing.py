import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# Step 1: Load Dataset
file_path = 'compas-scores.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Map 'score_text' to binary labels
df['binary_score'] = df['score_text'].map({'Low': 0, 'Medium': 0, 'High': 1})

# Step 3: Drop rows with null values in columns we are using
relevant_columns = ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'binary_score', 'sex', 'is_recid', 'r_charge_degree']
df_cleaned = df[relevant_columns].dropna()

# Step 4: Split Dataset
X = df_cleaned.drop(columns=['binary_score'])
y = df_cleaned['binary_score']

riskcount = y.value_counts()
risk1 = riskcount[0]
risk2 = riskcount[1]
print(risk1)
print(risk2)
print(X.shape)
# Convert categorical features to one-hot encoding and normalize numeric features
categorical_features = ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc', 'sex', 'is_recid', 'r_charge_degree']
continuous_features = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_categorical = encoder.fit_transform(X[categorical_features])
X_continuous = scaler.fit_transform(X[continuous_features].values.reshape(-1, len(continuous_features)))

X_processed = np.hstack([X_categorical, X_continuous])
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_processed_df = pd.DataFrame(X_processed, index=df_cleaned.index)

# Step 5: Logistic Regression Model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Define model, loss, and optimizer
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Step 6: Evaluate Model and Measure Bias
model.eval()
with torch.no_grad():
    y_pred_train = (model(X_train_tensor) >= 0.5).float()
    y_pred_test = (model(X_test_tensor) >= 0.5).float()

train_accuracy = accuracy_score(y_train, y_pred_train.numpy())
test_accuracy = accuracy_score(y_test, y_pred_test.numpy())
print(f"Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

def print_bias_analysis(y_true, y_pred, group_data, prefix=""):
    total_cases = len(y_true)
    predicted_positive = (y_pred == 1).sum()
    actual_negative = (y_true == 0).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    
    print(f"\n{prefix} Detailed Analysis for {group}:")
    print(f"Total cases: {total_cases}")
    print(f"Overall likelihood of recidivism prediction: {(predicted_positive/total_cases):.2%}")
    print(f"False positive rate (predicted recidivism for no-recidivism cases): {(false_positives/actual_negative):.2%}")
# Bias measurement: Equalized Odds
def equalized_odds(group, y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tpr = tp / (tp + fn + 1e-6)  # Avoid division by zero
    fpr = fp / (fp + tn + 1e-6)
    return tpr, fpr

# Iterate over each group and calculate Equalized Odds
grouped = df_cleaned.groupby('race')
for group, data in grouped:
    indices = data.index
    
    # Ensure y (target) and X (features) are aligned for the current group
    y_true = y.loc[indices]  # Use .loc to align indices
    X_group = X_processed_df.loc[indices]  # Ensure X_group is aligned with y_true
    
    # Generate predictions for the current group only
    model_input = torch.tensor(X_group.values, dtype=torch.float32)
    y_pred = (model(model_input).detach().numpy() >= 0.5).astype(int).flatten()  # Flatten to 1D array

    # Verify shape consistency
    assert len(y_true) == len(y_pred), "Mismatch between y_true and y_pred shapes"
    
    # Calculate Equalized Odds metrics
    tpr, fpr = equalized_odds(group, y_true, y_pred)
    print_bias_analysis(y_true, y_pred, group, "Initial Model")

# Step 7: Bias Mitigation with Re-sampling
# Resample to balance races

race_counts = df_cleaned['race'].value_counts()
min_count = race_counts.min()
balanced_df = pd.concat([df_cleaned[df_cleaned['race'] == race].sample(min_count, random_state=10) for race in df_cleaned['race'].unique()])

# Reset index of balanced_df to avoid indexing issues
balanced_df = balanced_df.reset_index(drop=True)

# Process features for balanced dataset
X_balanced = balanced_df.drop(columns=['binary_score'])
y_balanced = balanced_df['binary_score']

X_categorical_bal = encoder.transform(X_balanced[categorical_features])
X_continuous_bal = scaler.transform(X_balanced[continuous_features])
X_balanced_processed = np.hstack([X_categorical_bal, X_continuous_bal])

# Split the balanced dataset
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced_processed, y_balanced, test_size=0.2, random_state=42)

# Convert to tensor
X_train_bal_tensor = torch.tensor(X_train_bal, dtype=torch.float32)
y_train_bal_tensor = torch.tensor(y_train_bal.values, dtype=torch.float32).view(-1, 1)

# Retrain the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_bal_tensor)
    loss = criterion(outputs, y_train_bal_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Evaluate on balanced dataset
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
print("Equalized Odds for Balanced Dataset:")
for group, data in balanced_df.groupby('race'):
    # Get group data
    group_indices = data.index
    
    # Get predictions and true values for this group
    y_true = y_balanced.iloc[group_indices]
    y_pred = y_pred_bal[group_indices]
    
    # Calculate Equalized Odds metrics
    tpr, fpr = equalized_odds(group, y_true, y_pred)
    print_bias_analysis(y_true, y_pred, group, "Balanced Model")