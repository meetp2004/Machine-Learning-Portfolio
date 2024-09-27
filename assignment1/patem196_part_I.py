import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the GDP vs. happiness data from a CSV file.
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

# Clean the data for the year 2018
# We're filtering out unnecessary columns and keeping only the relevant data for analysis.
by_year = (data[data['Year'] == 2018]).drop(columns=["Continent", "Population (historical estimates)", "Code"]) 
# We also drop rows where the happiness score or GDP is missing.
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

# Create lists to hold GDP and happiness scores for countries with a happiness score above 4.5
happiness = []
gdp = []
for row in df.iterrows():
    # If the happiness score is above 4.5, we add it to our lists.
    if row[1]['Cantril ladder score'] > 4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])
        
class linear_regression():
    def __init__(self, x:list, y:list) -> None:
        # Initialize the regression model with input (GDP) and target (happiness) values.
        self.input = np.array(x)
        self.target = np.array(y)
    
    def preprocess(self):
        # Normalize the target (happiness) values for better model performance.
        hmean = np.mean(self.target)
        hstd = np.std(self.target)
        y_train = (self.target - hmean) / hstd

        # Normalize the input (GDP) values as well.
        gmean = np.mean(self.input)
        gstd = np.std(self.input)
        x_train = (self.input - gmean) / gstd
        
        # Create a design matrix X with a column of ones for the intercept.
        X = np.column_stack((np.ones(len(x_train)), x_train))

        # Prepare Y as a column vector.
        Y = (np.column_stack(y_train)).T
        return X, Y

    def train(self, X, Y, alpha, epochs):
        # Train the model using gradient descent.
        beta = np.random.randn(2, 1)  # Initialize beta coefficients randomly.
        n = len(self.input)
        for i in range(epochs):
            # Calculate the gradient of the loss function.
            gradient = 2 / n * (X.T).dot(X.dot(beta) - Y)
            # Update the beta coefficients.
            beta = beta - alpha * gradient
        return beta   
        
    def trainOls(self, X, Y):
        # Compute and return the beta coefficients using Ordinary Least Squares (OLS).
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) 
    
    def predict(self, X_test, beta):
        # Use the learned beta coefficients to predict values for new data.
        Y_hat = X_test.dot(beta)
        return Y_hat

    def mean_squared_error(self, ytrue, ypred):
        # Calculate the mean squared error between true and predicted values.
        return np.mean((ytrue - ypred) ** 2)

# Instantiate the linear regression class with GDP and happiness data.
lr_Gd = linear_regression(gdp, happiness)

# Preprocess the inputs to normalize them.
X, Y = lr_Gd.preprocess()

# Extract the normalized GDP values for plotting.
X_ = X[..., 1].ravel()

# Set up different learning rates and epochs to test for the best performance.
learning_rates = [0.001, 0.05, 0.5]
epochs = [125, 1000, 2500]

# Initialize variables to track the best results.
lowestErr = float('inf')
bestlr = 0
bestep = 0
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

# Loop through each combination of learning rates and epochs.
for lr in learning_rates:
    for ep in epochs:
        # Train the model with the current learning rate and number of epochs.
        beta = lr_Gd.train(X, Y, lr, ep)
        Y_predictGd = lr_Gd.predict(X, beta)
        mse = lr_Gd.mean_squared_error(Y, Y_predictGd)
        
        # Check if this is the best error so far, and update the best parameters if so.
        if mse < lowestErr:
            lowestErr = mse
            bestlr = lr
            bestep = ep
            bestBeta = beta
            
        # Print the training details for each run.
        print(f"LR: {lr}\t|\tEpochs: {ep}\t|\tBeta0: {beta[0][0]}\t|\tBeta1: {beta[1][0]}\t|\tMSE: {mse}")
        print('-------------------------------------------------------------------------------------------------------------------------------------')
        
        # Plot the regression line learned from gradient descent.
        axes[0].plot(X_, Y_predictGd, label=f'LR: {lr}, Epochs: {ep}') 
    print('\n')

# Now, let's perform OLS for comparison.
lr_ols = linear_regression(gdp, happiness)

# Compute the beta coefficients using OLS.
OlsBeta = lr_ols.trainOls(X, Y)

# Use the computed beta to make predictions.
Y_predictOls = lr_ols.predict(X, OlsBeta)
print(f"OlsBeta0: {OlsBeta[0][0]} \t|\t OlsBeta1: {OlsBeta[1][0]} \t|\t OlsMSE: {lr_ols.mean_squared_error(Y, Y_predictOls)}")
print('------------------------------------------------------------------------------------------')
print(f"Best Learning rate: {bestlr} \t|\t Best Epochs: {bestep} \t|\t Lowest MSE: {lowestErr}")

# Plot the original data points.
axes[0].scatter(X_, Y, color="red", label="Data points")
axes[1].scatter(X_, Y, color="red", label="Data points")

# Display the regression lines from both gradient descent and OLS.
axes[1].plot(X_, lr_Gd.predict(X, bestBeta), label='GD', color='g')
axes[1].plot(X_, Y_predictOls, label='OLS', color='r')

# Set the labels and titles for the plots.
axes[0].set_xlabel("Normalized GDP")
axes[0].set_ylabel("Normalized Happiness")
axes[0].set_title("Gradient Descent Experiment Regression Lines")
axes[1].set_xlabel("Normalized GDP")
axes[1].set_ylabel("Normalized Happiness")
axes[1].set_title("Best Gradient Descent vs Best OLS")
axes[1].legend()
axes[0].legend()
plt.legend()
plt.show()
