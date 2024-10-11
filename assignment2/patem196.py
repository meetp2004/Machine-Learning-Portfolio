# Import the necessary libraries for data manipulation, model building, and visualization
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math 

class SVM:
    
    def __init__(self, X, Y, C=1.0):
        # Initialize the SVM class with input features (X), labels (Y), and a regularization parameter C
        self.X = X
        self.Y = Y
        self.C = C
        # Set up the SVM model with an RBF kernel and automatic gamma value
        self.model = SVC(C=self.C, kernel="rbf", gamma='auto')
        self.X_ = None
        self.Y_ = None

    def feature_creation(self, feature_numbers):
        # This function creates new feature sets based on the input list of feature numbers
        if len(feature_numbers) > 4 or len(feature_numbers) < 1: return
        features = pd.DataFrame()
        # Concatenate feature subsets based on the feature number
        for feature_number in feature_numbers:
            if feature_number == 1:
                features = pd.concat([features, self.X.iloc[:, :18]], axis=1)
            elif feature_number == 2:
                features = pd.concat([features, self.X.iloc[:, 18:90]], axis=1)
            elif feature_number == 3:
                features = pd.concat([features, self.X.iloc[:, [90]]], axis=1)
            elif feature_number == 4:
                features = pd.concat([features, self.X.iloc[:, 91:]], axis=1)
            else:
                raise ValueError("Feature Number not Valid")
        # Update the input features with the selected feature subset
        self.X = features 

    def preprocess(self):
        # Convert the DataFrame to a numpy array for easier manipulation
        X_ = self.X.to_numpy()
        # Normalize the data by subtracting the mean and dividing by the standard deviation
        means = np.mean(X_, axis=0)
        std_devs = np.std(X_, axis=0)

        # Avoid division by zero by replacing 0 std deviations with 1
        std_devs[std_devs == 0] = 1

        # Standardize the data
        self.X_ = (X_ - means) / std_devs
        self.Y_ = self.Y.to_numpy()

    def train(self):
        # Train the SVM model using the preprocessed data
        self.model.fit(self.X_, self.Y_)

    def predict(self, X_test):
        # Use the trained SVM model to make predictions on test data
        return self.model.predict(X_test)

    def cross_validation(self, k, shuff, random):
        # Perform k-fold cross-validation on the data
        folds = KFold(n_splits=k, shuffle=shuff, random_state=random)
        accuracy = []
        tss = []
        matrices = []

        for train_index, test_index in folds.split(self.X_):
            # Split the data into training and testing sets for this fold
            X_train, X_test = self.X_[train_index], self.X_[test_index]
            Y_train, Y_test = self.Y_[train_index], self.Y_[test_index]

            # Train the model on the training data
            self.model.fit(X_train, Y_train)

            # Predict the labels of the test set
            Y_predicted = self.model.predict(X_test)

            # Calculate accuracy and TSS for this fold
            accuracy.append(accuracy_score(Y_test, Y_predicted))
            tss.append(self.tss(Y_test, Y_predicted))

            # Store the confusion matrix for later use
            matrices.append(confusion_matrix(Y_test, Y_predicted))

        # Return the average accuracy, TSS scores, and confusion matrices across all folds
        return np.mean(accuracy), tss, matrices
    
    def tss(self, Y_true, Y_pred):
        # Calculate the confusion matrix and unpack its values
        cm = confusion_matrix(Y_true, Y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = fp / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate TSS (True Skill Statistic)
        return sensitivity - specificity
  
# This function helps convert flare class labels to binary form (-1 or 1)
def classify_flare(flare_class):
    return -1 if flare_class is None else 1

# Utility function to plot a confusion matrix for a specific feature combination
def get_confusion_matrix(confusion_matrices, combination):
    # Sum confusion matrices from all folds to get a combined matrix
    combined_conf_matrix = np.sum(confusion_matrices, axis=0)
    
    # Create a heatmap of the confusion matrix using seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    # Add title and labels for better understanding
    plt.title(f'Combined Confusion Matrix of {combination}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Display the confusion matrix plot
    plt.tight_layout()
    plt.show()

# Plot TSS scores for all combinations across all folds
def plot_all_tss(all_tss_scores, all_combinations):
    plt.figure(figsize=(10, 6))
    
    # Plot the TSS scores for each feature combination on the same graph
    for idx, combination in enumerate(all_combinations):
        plt.plot(range(1, 11), all_tss_scores[idx], marker='o', linestyle='-', label=f"Combination {combination}")
    
    # Add plot title, axis labels, and grid for clarity
    plt.title("TSS Scores for All Combinations")
    plt.xlabel("Fold Number")
    plt.ylabel("TSS Score")
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--')
    plt.legend(title="Combinations")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to generate all possible combinations of feature sets
def power_set(s):
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))

# Define the directories for datasets
directories = ['./data-2010-15/data-2010-15', './data-2020-24/data-2020-24']

# Function to experiment with different feature sets
def feature_experiment(path):
    directory = path

    # Load the various feature sets and labels from the data files
    fs_ii_pos = np.load(f"{directory}/pos_features_main_timechange.npy", allow_pickle=True)
    fs_ii_neg = np.load(f"{directory}/neg_features_main_timechange.npy", allow_pickle=True)

    fs_iii_pos = np.load(f"{directory}/pos_features_historical.npy", allow_pickle=True)
    fs_iii_neg = np.load(f"{directory}/neg_features_historical.npy", allow_pickle=True)

    fs_iiii_pos = np.load(f"{directory}/pos_features_maxmin.npy", allow_pickle=True)
    fs_iiii_neg = np.load(f"{directory}/neg_features_maxmin.npy", allow_pickle=True)

    labels_pos = np.load(f"{directory}/pos_class.npy", allow_pickle=True)
    labels_neg = np.load(f"{directory}/neg_class.npy", allow_pickle=True)

    # Combine the positive and negative features into one dataset
    fs_pos = np.column_stack((fs_ii_pos, fs_iii_pos, fs_iiii_pos))
    fs_neg = np.column_stack((fs_ii_neg, fs_iii_neg, fs_iiii_neg))

    # Create DataFrames for the positive and negative features
    df_pos = pd.DataFrame(fs_pos, columns=[f'FS_Feature_{i + 1}' for i in range(fs_pos.shape[1])])
    df_pos['FLARE'] = [classify_flare(label[2]) for label in labels_pos]

    df_neg = pd.DataFrame(fs_neg, columns=[f'FS_Feature_{i + 1}' for i in range(fs_neg.shape[1])])
    df_neg['FLARE'] = [classify_flare(label[2]) for label in labels_neg]

    # Combine the positive and negative data into one DataFrame
    df_combined = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

    X = df_combined.iloc[:, :-1]  # Features
    Y = df_combined.iloc[:, -1]   # Labels
    
    # Generate all possible combinations of feature sets
    all_combinations = [list(tup) for tup in power_set([1,2,3,4])[1:]]
    all_tss_scores = []
    best_tss = 0
    best_combination = []

    # Iterate over each combination and evaluate the model
    for combination in all_combinations:
        my_svm = SVM(X, Y, C=1)
        my_svm.feature_creation(combination)
        my_svm.preprocess()
        accuracy, tss_scores, all_matrices = my_svm.cross_validation(10, True, 42)

        # Track the best feature set based on the average TSS score
        mean_tss = np.mean(tss_scores)
        if mean_tss > best_tss:
            best_tss = mean_tss
            best_combination = combination

        # Store TSS scores for each combination
        all_tss_scores.append(tss_scores)

        # Visualize the confusion matrix for the best combination
        get_confusion_matrix(all_matrices, combination)
        print(f"Combination {combination}:\n\tAverage Accuracy: {accuracy}\n\tAverage TSS Score: {mean_tss}\n")
    # Plot TSS scores for all combinations
    plot_all_tss(all_tss_scores, all_combinations)

    # Print the best TSS and its corresponding feature combination
    print(f"Best TSS score: {best_tss} with combination: {best_combination}")
    return best_combination
def data_experiment(best, pathList):
    for directory in pathList:

        fs_ii_pos = np.load(f"{directory}/pos_features_main_timechange.npy", allow_pickle=True)
        fs_ii_neg = np.load(f"{directory}/neg_features_main_timechange.npy", allow_pickle=True)

        fs_iii_pos = np.load(f"{directory}/pos_features_historical.npy", allow_pickle=True)
        fs_iii_neg = np.load(f"{directory}/neg_features_historical.npy", allow_pickle=True)

        fs_iiii_pos = np.load(f"{directory}/pos_features_maxmin.npy", allow_pickle=True)
        fs_iiii_neg = np.load(f"{directory}/neg_features_maxmin.npy", allow_pickle=True)

        labels_pos = np.load(f"{directory}/pos_class.npy", allow_pickle=True)
        labels_neg = np.load(f"{directory}/neg_class.npy", allow_pickle=True)

        # Concatenate the positive and negative features
        fs_pos = np.column_stack((fs_ii_pos, fs_iii_pos, fs_iiii_pos))
        fs_neg = np.column_stack((fs_ii_neg, fs_iii_neg, fs_iiii_neg))

        # Create the Positive DataFrame
        df_pos = pd.DataFrame(fs_pos, columns=[f'FS_Feature_{i + 1}' for i in range(fs_pos.shape[1])])
        df_pos['FLARE'] = [classify_flare(label[2]) for label in labels_pos]

        # Create the Negative DataFrame
        df_neg = pd.DataFrame(fs_neg, columns=[f'FS_Feature_{i + 1}' for i in range(fs_neg.shape[1])])
        df_neg['FLARE'] = [classify_flare(label[2]) for label in labels_neg]

        # Combine the data
        df_combined = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

        X = df_combined.iloc[:, :-1]
        Y = df_combined.iloc[:, -1]

        # Use the best feature set from the previous experiment
        best_feature_set = best  # Replace with your actual best feature set from feature_experiment

        my_svm = SVM(X, Y, C=1)
        my_svm.feature_creation(best_feature_set)
        my_svm.preprocess()
        
        accuracy, tss_scores, all_matrices = my_svm.cross_validation(10, True, 22)

        # Plot the confusion matrix for the current dataset
        get_confusion_matrix(all_matrices, f"Dataset: {directory}")

        # Store TSS scores for line plotting
        if 'all_tss_scores' not in locals():
            all_tss_scores = []
        all_tss_scores.append(tss_scores)

    # Plot TSS scores for both datasets on the same line graph
    plt.figure(figsize=(10, 6))
    for idx, tss in enumerate(all_tss_scores):
        plt.plot(range(1, 11), tss, marker='o', linestyle='-', label=f"Dataset {idx + 1}")

    # Add title and labels
    plt.title("TSS Scores for Both Datasets")
    plt.xlabel("Fold Number")
    plt.ylabel("TSS Score")

    # Set ticks and limits for the y-axis
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)

    # Add grid and horizontal line at y=0
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--')

    # Add a legend to distinguish between different datasets
    plt.legend(title="Datasets")

    # Display the plot
    plt.tight_layout()
    plt.show()

def no_shuffle_experiment(path):
    directory = path
    # Load the data order from the .npy file
    data_order = np.load(f"{directory}/data_order.npy", allow_pickle=True)

    fs_ii_pos = np.load(f"{directory}/pos_features_main_timechange.npy", allow_pickle=True)
    fs_ii_neg = np.load(f"{directory}/neg_features_main_timechange.npy", allow_pickle=True)

    fs_iii_pos = np.load(f"{directory}/pos_features_historical.npy", allow_pickle=True)
    fs_iii_neg = np.load(f"{directory}/neg_features_historical.npy", allow_pickle=True)

    fs_iiii_pos = np.load(f"{directory}/pos_features_maxmin.npy", allow_pickle=True)
    fs_iiii_neg = np.load(f"{directory}/neg_features_maxmin.npy", allow_pickle=True)

    labels_pos = np.load(f"{directory}/pos_class.npy", allow_pickle=True)
    labels_neg = np.load(f"{directory}/neg_class.npy", allow_pickle=True)

    # Concatenate the positive and negative features
    fs_pos = np.column_stack((fs_ii_pos, fs_iii_pos, fs_iiii_pos))
    fs_neg = np.column_stack((fs_ii_neg, fs_iii_neg, fs_iiii_neg))

    # Create the Positive DataFrame
    df_pos = pd.DataFrame(fs_pos, columns=[f'FS_Feature_{i + 1}' for i in range(fs_pos.shape[1])])
    df_pos['FLARE'] = [classify_flare(label[2]) for label in labels_pos]

    # Create the Negative DataFrame
    df_neg = pd.DataFrame(fs_neg, columns=[f'FS_Feature_{i + 1}' for i in range(fs_neg.shape[1])])
    df_neg['FLARE'] = [classify_flare(label[2]) for label in labels_neg]

    # Combine the data
    df_combined = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

    # Reorder the data based on the order provided in the data_order.npy file
    df_combined = df_combined.iloc[data_order]

    # Split into features (X) and target labels (Y)
    X = df_combined.iloc[:, :-1]
    Y = df_combined.iloc[:, -1]

    # Select the best feature set from the previous experiment
    best_feature_set = [1, 4]  # Replace with the actual best combination
    my_svm = SVM(X, Y, C=1)
    my_svm.feature_creation(best_feature_set)
    my_svm.preprocess()

    # Perform cross-validation without shuffling the data (shuffle=False)
    accuracy, tss_scores, all_matrices = my_svm.cross_validation(10, False, None)
    # Plot the confusion matrix for the no-shuffle experiment
    get_confusion_matrix(all_matrices, "No Shuffle Experiment")
    print(f" No-Shuffle Accuracy: {accuracy}, No-Shuffle TSS: {np.mean(tss_scores)}")
    

best = feature_experiment(directories[0])
data_experiment(best, directories)
no_shuffle_experiment(directories[0])