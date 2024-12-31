# Author: Swati Mishra
# Due Date: November 10th
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVMClassifier:
    
    def __init__(self, lr, n_epochs, penalty_factor, X_train, Y_train):
        # Initialize parameters for the SVM classifier
        self.X_data = X_train
        self.Y_data = Y_train
        self.lr = lr
        self.n_epochs = n_epochs
        self.penalty_factor = penalty_factor
        self.scaled_X = self.X_data
        self.scaled_Y = self.Y_data
        self.weights = np.random.randn(X.shape[1])
        self.early_stop_weights = None

    def normalize_data(self):
        # Standardizing the training data
        scaler = StandardScaler().fit(self.X_data)
        self.scaled_X = scaler.transform(self.X_data)
        self.scaled_Y = self.Y_data

    def _calculate_gradient (self, X, Y):
        X_ = np.array([X])
        hinge_distance = 1 - (Y * np.dot(X_, self.weights))
        total_distance = np.zeros(len(self.weights))
        return total_distance + self.weights if max(0, hinge_distance[0]) == 0 else total_distance + self.weights - (self.penalty_factor * Y * X_[0])
    
    # Computes the Hinge Loss given X and Y data
    def _calculate_loss (self, X, Y):
        
        loss = 0
        regularization = 0.5 * np.linalg.norm(self.weights) ** 2
        for i in range(X.shape[0]):
            temp = np.dot(X[i], self.weights)
            hinge = 1 - Y[i] * temp 
            loss += max(0, hinge)
            
        return self.penalty_factor * loss + regularization

    def train_with_sgd(self, X_test, Y_test, tol, stop_early = True):
        
        last_loss = float('inf') 
        delta_loss = tol 
        
        # record training and test loss
        train_loss_dict = dict()
        test_loss_dict = dict()
        early_stopped = 0 
        
        not_stopped = True # boolean flag
        temp = 0 # Tracks epochs in case of not stopping

        for epoch in range(self.n_epochs):

            features, output = shuffle(self.scaled_X, self.scaled_Y)
            
            for i, feature in enumerate(features):
                gradient = self._calculate_gradient(feature, output[i])
                self.weights = self.weights - (self.lr * gradient)
            
            current_train_loss = self._calculate_loss(features, output) 
            current_val_loss = self._calculate_loss(X_test, Y_test)
                        
            # Once we've reached 1/10 of epochs, we report the loss and add for plotting
            if epoch%(self.n_epochs//10)==0:
                
                print(f"Epoch is: {epoch} | Training Loss is: {current_train_loss} | Testing Loss is: {current_val_loss}")
                train_loss_dict[epoch] = current_train_loss
                test_loss_dict[epoch] = current_val_loss
                if not_stopped:
                    temp = epoch 

            # If it early stops, we store the weights to test performance later
            if early_stopped == 0 and abs(last_loss - current_train_loss) <= delta_loss and stop_early:
                print("Early Stopping on Epoch:", epoch)
                early_stopped = epoch
                not_stopped = False  
                self.stop_weights = self.weights
            
            last_loss = current_train_loss
            
        # Never Stopped
        if early_stopped == 0:
            early_stopped = temp
        
        print("Training ended...")
        
        return train_loss_dict, test_loss_dict, early_stopped

    def mini_batch_sgd(self, batch_size, X_val, Y_val):
        train_loss_dict = {}
        val_loss_dict = {}

        for epoch in range(self.n_epochs):
            shuffled_X, shuffled_Y = shuffle(self.scaled_X, self.scaled_Y)

            for start_idx in range(0, len(shuffled_X), batch_size):
                end_idx = min(start_idx + batch_size, len(shuffled_X))
                batch_X = shuffled_X[start_idx:end_idx]
                batch_Y = shuffled_Y[start_idx:end_idx]
                avg_gradient = np.mean([self._calculate_gradient(batch_X[i], batch_Y[i]) for i in range(len(batch_X))], axis=0)
                self.weights -= self.lr * avg_gradient

            train_loss = self._calculate_loss(shuffled_X, shuffled_Y)
            val_loss = self._calculate_loss(X_val, Y_val)

            if epoch % (self.n_epochs // 10) == 0:
                print(f"Epoch {epoch}: Training Loss={train_loss}, Validation Loss={val_loss}")
                train_loss_dict[epoch] = train_loss
                val_loss_dict[epoch] = val_loss

        return train_loss_dict, val_loss_dict

    def predict(self, X):
        # Predict class labels based on the learned weights
        return np.sign(np.dot(X, self.weights))

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = accuracy_score(Y, predictions)
        precision = precision_score(Y, predictions)
        recall = recall_score(Y, predictions)
        return accuracy, precision, recall
    
    # Sampling strategy as required, selecting the min error
    def sampling_strategy(self,X_unlabelled,Y_unlabelled):
        losses = [self._calculate_loss(np.array([x]), np.array([y])).item() for x, y in zip(X_unlabelled, Y_unlabelled)]
        min_loss_index = np.argmin(losses)
        return min_loss_index
    
# Part 1: We make an SVM model training using Stochastic Gradient Descent with Early Stopping
def part_1 (X_train, Y_train, X_test, Y_test, C, learning_rate, epochs):
    
    # Initialize the model and preprocess
    svm = SVMClassifier(learning_rate, epochs, C, X_train, Y_train)
    svm.normalize_data()
    
    # Train the model 
    train_loss, test_loss, early_stopped_epochs = svm.train_with_sgd(X_test, Y_test, 0.001)
    
    # plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(list(train_loss.keys()), list(train_loss.values()), label='Training Loss', color="g")
    plt.plot(list(test_loss.keys()), list(test_loss.values()), label='Validation Loss', color="r")
    plt.axvline(x=early_stopped_epochs, color='r', linestyle='--', label='Early Stopped')
    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs (SGD with Early Stopping)')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    
    return train_loss, test_loss, svm

# SVM model trained using Mini-Batch SGD with NO early stopping.
def part_2 (X_train, Y_train, X_test, Y_test, part1_train, part1_test, C, learning_rate, epochs):
    
    # Initialize the model and preprocess
    svm = SVMClassifier(learning_rate, epochs, C, X_train, Y_train)
    svm.normalize_data()
        
    # obtain the dictionaries
    train_loss, test_loss = svm.mini_batch_sgd(10, X_test, Y_test)
    
    # plot the losses (compares SGD vs Mini-Batch SGD)
    plt.figure(figsize=(10, 6))
    plt.plot(list(train_loss.keys()), list(train_loss.values()), label='Training Loss for SGD', color="b")
    plt.plot(list(test_loss.keys()), list(test_loss.values()), label='Validation Loss for SGD', color="b", linestyle='--')
    plt.plot(list(part1_train.keys()), list(part1_train.values()), label='Training Loss for Mini-Batch SGD', color="g")
    plt.plot(list(part1_test.keys()), list(part1_test.values()), label='Validation Loss for Min-Batch SGD', color="g", linestyle='--')
    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs (SGD vs Mini-Batch SGD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return svm

# Part 3: Use the Active Learning Strategy from the assignment
def part_3(X, Y, C, learning_rate, training_epochs, delta_loss, max_samples):
    last_loss = float('inf') 
    samples_used = initial_samples = 10
    
    # Randomly get our initial samples \
    random_indices = np.random.choice(len(X), initial_samples, replace=False)
    X_initial_train = X[random_indices]
    Y_initial_train = Y[random_indices]
    X_unlabelled = np.delete(X, random_indices, axis=0)
    Y_unlabelled = np.delete(Y, random_indices, axis=0)
    
    # initialize the model and preprocess
    svm = SVMClassifier(learning_rate, training_epochs, C, X_initial_train, Y_initial_train)
    svm.normalize_data()

    # Set our dictionaries
    train_loss_dict = dict()
    test_loss_dict = dict()
    
    # Stopping Condition 
    while samples_used < max_samples and len(X_unlabelled) > 0:

        # Normalizes the unlabelled data 
        scaler = StandardScaler().fit(X_unlabelled)
        X_unlabelled_norm = scaler.transform(X_unlabelled) # we keep track of the old unnormalized unlabelled data
        
        
        current_train_loss_dict, current_test_loss_dict, _ = svm.train_with_sgd(X_unlabelled_norm, Y_unlabelled, 1e-3, stop_early=False)
        current_train_loss = svm._calculate_loss(svm.scaled_X, svm.scaled_Y)
        
        # If we've converged, we set the training and tess loss dictionaries to the current dictionaries
        if abs(last_loss - current_train_loss) <= delta_loss:
            print(f"Converges with {samples_used} training samples\n")
            test_loss_dict = current_test_loss_dict
            train_loss_dict = current_train_loss_dict
            break
        
        last_loss = current_train_loss

        # add it to the training data and preprocess
        index = svm.sampling_strategy(X_unlabelled_norm, Y_unlabelled)
        svm.X_data = np.vstack((svm.X_data, [X_unlabelled[index]]))
        svm.Y_data = np.append(svm.Y_data, Y_unlabelled[index])
        svm.normalize_data()    
        
        # remove the chosen sample
        X_unlabelled = np.delete(X_unlabelled, index, axis=0)
        Y_unlabelled = np.delete(Y_unlabelled, index, axis=0)
        samples_used += 1
        print(samples_used)
        
    # plots the losses
    plt.figure(figsize=(10, 6))
    plt.plot(list(train_loss_dict.keys()), list(train_loss_dict.values()), label='Training Loss', color="b")
    plt.plot(list(test_loss_dict.keys()), list(test_loss_dict.values()), label='Validation Loss', color="g")
    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Best Training and Validation Loss with {samples_used} Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

    return samples_used, svm
    
np.random.seed(40)

data = pd.read_csv("data1.csv")

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

X = data.iloc[:, 1:]
X.insert(loc = len(X.columns), column="bias", value=1)
X_features = X.to_numpy()

category_dict = {"B" : -1.0, "M" : 1.0}
Y = np.array([(data.loc[:, "diagnosis"]).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)
scalarTest = StandardScaler().fit(X_test)
X_test = scalarTest.transform(X_test)

print("Part 1 Train\n")
p1_train, p1_test, part1_svm = part_1(X_train, Y_train, X_test, Y_test, 0.01, 0.001, 100)

print("Part 2 Train\n")
part2_svm = part_2(X_train, Y_train, X_test, Y_test, p1_train, p1_test, 0.01, 0.001, 100)

print("Part 3 Train\n")
part3 = part_3(X_train, Y_train, 0.01, 0.05, 100, 0.01, max_samples=len(X_train)) #Set the max sample to the length of the training set

# Part 1
print("Part 1 Performance")
train_metrics = part1_svm.evaluate(part1_svm.scaled_X, part1_svm.scaled_Y)
test_metrics = part1_svm.evaluate(X_test, Y_test)
print(f"Training: [Accuracy : {train_metrics[0]} / Precision : {train_metrics[1]} / Recall : {train_metrics[2]}]")
print(f"Testing: [Accuracy : {test_metrics[0]} / Precision : {test_metrics[1]} / Recall : {test_metrics[2]}]\n")

#Part 2
print("Part 2 Performance")
train_metrics = part2_svm.evaluate(part2_svm.scaled_X, part2_svm.scaled_Y)
test_metrics = part2_svm.evaluate(X_test, Y_test)
print(f"Training: [Accuracy : {train_metrics[0]} / Precision : {train_metrics[1]} / Recall : {train_metrics[2]}]")
print(f"Testing: [Accuracy : {test_metrics[0]} / Precision : {test_metrics[1]} / Recall : {test_metrics[2]}]\n")

# Part 3
part3_svm = part3[1]
print("Part 3 Performance")
print(f"Samples Used: {part3[0]}\n")
train_metrics = part3_svm.evaluate(part3_svm.scaled_X, part3_svm.scaled_Y) 
test_metrics = part3_svm.evaluate(X_test, Y_test)
print(f"Training: [Accuracy : {train_metrics[0]} / Precision : {train_metrics[1]} / Recall : {train_metrics[2]}]")
print(f"Testing: [Accuracy : {test_metrics[0]} / Precision : {test_metrics[1]} / Recall : {test_metrics[2]}]\n")