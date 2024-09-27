import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
class LinearRegression:
    
    def __init__ (self, x, y):
        self.X = x
        self.Y = y
        
    def preprocess (self):
        
        # Normalizing Targets
        happ_mean = np.mean(self.Y)
        happ_std = np.std(self.Y)
        y_train = (self.Y - happ_mean) / happ_std
        Y = (np.column_stack(y_train)).T
        
        #Normalizing Inputs
        gdp_mean = np.mean(self.X)
        gdp_std = np.std(self.X)
        x_train = (self.X - gdp_mean) / gdp_std
        X = np.column_stack((np.ones(len(x_train)),x_train)) # Arrange in Matrix Format
        
        return X, Y

    def train_ols (self, X, Y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def train_gd (self, learning_rate, epochs : int, X, Y):
        n = len(X)
        beta = np.random.randn(2, 1)
        for _ in range (epochs):
            gradient = 2 / n * (X.T).dot(X.dot(beta) - Y)
            beta = beta - learning_rate * gradient
        return beta
        
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test * beta.T
        return np.sum(Y_hat,axis=1)
    
    def mean_squared_error (self, Y_true, Y_pred):
        return np.mean((Y_true - Y_pred) ** 2)
    
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

# Remove unnecessary columns and rows
data = data.query("Year == 2018")
data = data[["Cantril ladder score", "GDP per capita, PPP (constant 2017 international $)"]]

# Rename the columns
data.rename(columns = {"Cantril ladder score": "Happiness", "GDP per capita, PPP (constant 2017 international $)": "GDP Per Capita"}, inplace=True)
data = data.dropna()

#Gets both X and Y Lists
happiness = data["Happiness"]
gdp = data["GDP Per Capita"]
    
linear_regression_model = LinearRegression(gdp, happiness)
X, Y = linear_regression_model.preprocess() #preprocess the data

#Get the parameters based on OLS
beta_ols = linear_regression_model.train_ols(X, Y)

# access the 1st column (the 0th column is all 1's)
X_ = X[..., 1].ravel() 

fig, ax = plt.subplots(1, 2, figsize = (20, 8))

#Labelling the Plots
ax[0].set_title("Linear Regression using Ordinary Least Squares")
ax[1].set_title("Linear Regression optimized with Gradient Descent")

ax[0].set_xlabel("GDP Per Capita ($)")
ax[0].set_ylabel("Happiness (Cantril Ladder Score)")

ax[1].set_xlabel("GDP Per Capita ($)")
ax[1].set_ylabel("Happiness (Cantril Ladder Score)")

#Plot the OLS Fitted Line
ax[0].scatter(X_, Y, color='black', label="Data Points")
ax[1].scatter(X_, Y, color='black', label="Data Points")

#Plot the OLS Fitted Line
Y_prediction_ols = X.dot(beta_ols)
error_ols = linear_regression_model.mean_squared_error(Y, Y_prediction_ols)
ax[0].plot(X_, Y_prediction_ols, color='red', label="Line of Best Fit for OLS")

ax[0].legend()

#Plot the Gradient Descent Fitted Lines
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
epochs_list = [125, 500, 1000, 2000, 2500]

best_error = float('inf')
best_beta = None
best_learning_rate = None
best_epoch = None

for rate in learning_rates:
    for epoch in epochs_list:
        beta_gd = linear_regression_model.train_gd(rate, epoch, X, Y)
        Y_prediction_gd = X.dot(beta_gd)
        error_gd = linear_regression_model.mean_squared_error(Y, Y_prediction_gd)
        if error_gd < best_error:
            best_error = error_gd
            best_beta = beta_gd
            best_learning_rate = rate
            best_epoch = epoch
        
        ax[1].plot(X_, Y_prediction_gd, label=f"Learning Rate: ${rate} | Epochs: ${epoch}") #Labels each line with the number of epochs and learning rate
        print(f"Learning Rate: {rate} | Epochs: {epoch} | Beta_0': {beta_gd[0]} | Beta_1': {beta_gd[1]} | Error : {error_gd}") #Prints out the corresponding beta' values
        
print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best Epochs: {best_epoch}")
print(f"Best Error: {best_error}")
print(f"Best Beta: {best_beta}")
print(f"Best Beta_OLS: {beta_ols}")
        
ax[1].legend(prop={'size': 7})
plt.tight_layout()
plt.show()

plt.scatter(X_, Y, color='black')
plt.plot(X_, X.dot(best_beta), label="GD line")
plt.plot(X_, X.dot(beta_ols), label="OLS Line")
plt.legend()
plt.show()