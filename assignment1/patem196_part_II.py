import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MultiLinearRegression:
    
    def __init__(self, X, Y) -> None:
        # Here we're initializing our model with the input features (X) and target variable (Y).
        self.input = X
        self.target = Y
        
    def preprocess(self) -> None:
        # Let's standardize our input features and target variable
        # This helps in normalizing the data, making our model training more effective.
        self.input = (self.input - np.mean(self.input, axis=0)) / np.std(self.input, axis=0)
        self.target = (self.target - np.mean(self.target)) / np.std(self.target)
        
        # Now, we create our design matrix X by adding a column of ones for the intercept term.
        X = np.column_stack((np.ones(len(self.input)), self.input))
        # We also arrange our target variable Y into a column vector format.
        Y = (np.column_stack(self.target)).T
        
        return X, Y
        
    def k_folds_gen(self, k: int) -> list:
        # This function generates k folds for cross-validation.
        k_folds_list = []
        length = len(self.input) // k
        
        for i in range(k):
            # Here, we define our test data for the current fold.
            test_x = self.input.iloc[i * length: (i + 1) * length, :]
            test_y = self.target.iloc[i * length: (i + 1) * length]
            
            # Now we select our training data by dropping the current test fold.
            train_x = self.input.drop(self.input.index[i * length: (i + 1) * length])
            train_y = self.target.drop(self.target.index[i * length: (i + 1) * length])
        
            # We append our test and train sets to the list for later use.
            k_folds_list.append([[test_x, test_y], [train_x, train_y]])
        
        return k_folds_list
    
    def predict(self, X, beta):
        # This function predicts the target values using the linear model.
        return X.dot(beta)
    
    def train_ols(self, X, Y):
        # Here, we're training our model using Ordinary Least Squares (OLS).
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def mean_squared_error(self, Y_true, Y_pred):
        # This function calculates the mean squared error, a common metric to evaluate model performance.
        return np.mean((Y_true - Y_pred) ** 2)

    def k_fold(self, k_list):
        # In this method, we perform k-fold cross-validation and calculate the average mean squared error.
        mse = 0
        for i in range(len(k_list)):
            # Grab the test and train sets from the k-fold list.
            test = k_list[i][0]
            train = k_list[i][1]
            
            # Create and preprocess the model for the training set.
            train_model = MultiLinearRegression(train[0], train[1])
            X_Train, Y_Train = train_model.preprocess()
            
            # Create and preprocess the model for the test set.
            test_model = MultiLinearRegression(test[0], test[1])
            X_Test, Y_Test = test_model.preprocess()
                
            # Train the model and get the OLS coefficients (betas).
            beta_ols = train_model.train_ols(X_Train, Y_Train)
                    
            # Use the trained model to make predictions on the test set.
            Y_predict_ols = test_model.predict(X_Test, beta_ols)
            
            # Accumulate the mean squared error for this fold.
            mse += self.mean_squared_error(Y_Test, Y_predict_ols)
        
        # Return the average MSE across all folds.
        return mse / len(k_list)    
    
# Set the random seed for reproducibility.
np.random.seed(42)            

# Load the data from a CSV file.
data = pd.read_csv("datasets/training_data.csv")
# Select relevant columns for analysis.
data = data[["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]]
# Adjust the target variable 'Rings' by adding 1.5.
data["Rings"] = data["Rings"] + 1.5

# Separate the input features (X) from the target variable (Y).
X_in = data.loc[:, data.columns != "Rings"]
Y_in = data.loc[:, "Rings"]

# Initialize our multi-linear regression model.
final_model = MultiLinearRegression(X_in, Y_in)
X, Y = final_model.preprocess()

# Store the preprocessed input and target.
X = final_model.input
Y = final_model.target

# Generate k-folds for cross-validation.
k_list = final_model.k_folds_gen(5)
print("Cross Validation Average MSE:", final_model.k_fold(k_list))

# Train the model on the entire dataset to get the coefficients. 
# This is done to ensure we are using all our data for training
betas = final_model.train_ols(X, Y)
# Make predictions using the trained model.
Y_predict = final_model.predict(X, betas)

# Visualize the results with scatter plots.
fig, ax = plt.subplots(3, 3, figsize=(20, 8))

# Create scatter plots for each feature against the target variable.
for i, value in enumerate(X.columns):
    row = i % 3 
    col = i // 3  

    ax[row, col].set_title(X.columns[i] + " vs Age(Years)")
    ax[row, col].set_xlabel(X.columns[i])
    ax[row, col].set_ylabel("Age(Years)")
    ax[row, col].scatter(X[value].tolist(), Y, color='blue', label='Original Scatter')
    ax[row, col].scatter(X[value].tolist(), Y_predict, color='red', alpha=0.30, label='Predicted Values')
    
    # Add a legend
    ax[row, col].legend()

plt.tight_layout()    
plt.show()
