# Description: This script compares supervised three classifiers (Random Forest, K-Nearest Neighbors, and Support Vector Classifier)
#              on the digits dataset using cross-validation. also optimises them by tuning hyperparameters for each classifier 
#              and output plots results
# Created on: [20/11/2024]
# --------------------------------------------------
# Dependencies: Requires numpy, pandas, sklearn, and matplotlib
# Inputs: Digits dataset
# Outputs: accuracy, standard deviation for each model along with plots, confusion matrices

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC



#Student Number to seed
np.random.seed(38)

#load dataset and Split data into training and testing sets 85/15
data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=38)




#Cross validation Function 
def CrossValidation(X, y,n_splits=5):
 
    fold_size = len(X) // 5
    
    results = []
    
    # Loop through each fold 
    for i in range(n_splits):
       
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_splits - 1 else len(X)
        
        val_indices = np.arange(start_idx, end_idx)
        
        train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
        results.append((train_indices, val_indices))
    
    return results


# function to calc mean accuracy  and standard deviation uses previous function 
def evaluate_classifier(classifier, X_train, y_train, n_splits=5):
    cv_splits = CrossValidation(X_train, y_train, n_splits=n_splits)
    fold_accuracies = []
    
    for train_indices, val_indices in cv_splits:
        #split data into current fold's train and validation sets
        X_fold_train, X_fold_val = X_train[train_indices], X_train[val_indices]
        y_fold_train, y_fold_val = y_train[train_indices], y_train[val_indices]
        
        # train classifier and evaluate accuracy on validation set
        classifier.fit(X_fold_train, y_fold_train)
        y_pred = classifier.predict(X_fold_val)
        fold_accuracies.append(accuracy_score(y_fold_val, y_pred))
    
    # Return mean and standard deviation of accuracies across folds
    return np.mean(fold_accuracies), np.std(fold_accuracies)



#Run each classifier as default
classifiers = {
    "Random Forest (Default)": RandomForestClassifier(random_state=38),
    "K-Nearest Neighbors (Default)": KNeighborsClassifier(),
    "SVC (Default)": SVC(random_state=38)
}
# Create Dataframe to store the results
default_results = []

for name, clf in classifiers.items():
    mean_accuracy, std_accuracy = evaluate_classifier(clf, X_train, y_train)
    default_results.append({
        'Classifier': name,
        'Mean Accuracy': mean_accuracy,
        'Standard Deviation': std_accuracy
    })

# Convert results to a DataFrame
default_results_df = pd.DataFrame(default_results)
print('\n')
print("Default Classifier Performance:")
print(default_results_df)




knn_configs = [
    #values 2-7 for k
    {'n_neighbors': k} for k in range(2, 8) 
]

knn_results = []
for config in knn_configs:
    knn = KNeighborsClassifier(n_neighbors=config['n_neighbors'])
    accuracies = []

    ## Calculate accuracy and standard deviation for each k value
    for train_idx, val_idx in CrossValidation(X_train, y_train, n_splits=5):
        knn.fit(X_train[train_idx], y_train[train_idx])  
        val_pred = knn.predict(X_train[val_idx])        
        accuracies.append(accuracy_score(y_train[val_idx], val_pred))

    #calculate mean
    mean_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)
    knn_results.append({'K': config['n_neighbors'], 'Mean Accuracy': mean_accuracy, 'Standard Deviation': std_dev_accuracy})

# create Dataframe for KNN results
knn_optimization_df = pd.DataFrame(knn_results)
print("\nKNN Optimization Results:\n", knn_optimization_df)



#define parameter values to test for Random Forest and result list
n_estimators_values = [ 100, 150, 200, 300]  
max_depth_values = [5, 10, 15, 20,30]       

rf_optimization_results = []

# loop for each combination of n_estimators and max_depth and train on each combination
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=38)
        
        # run classfier function
        mean_accuracy, std_deviation = evaluate_classifier(rf, X_train, y_train)
        
        #add to results to list
        rf_optimization_results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'Mean Accuracy': mean_accuracy,
            'Standard Deviation': std_deviation
        })

# convert results to a dataframe
rf_optimization_df = pd.DataFrame(rf_optimization_results)
print("\nRandom Forest Optimization Results:")
print(rf_optimization_df)



# SVC Test different kernels and gamma values
kernel_types = ['rbf', 'poly']
gamma_values = [0.001, 0.01, 0.1, 1, 10]  
svc_optimization_results = []

#loop through kernel and gamma list and trains a model with each combination
for kernel in kernel_types:
    for gamma in gamma_values:
        svc = SVC(kernel=kernel, gamma=gamma, random_state=38)
        mean_accuracy, std_deviation = evaluate_classifier(svc, X_train, y_train)
        svc_optimization_results.append({
            'Kernel': kernel,
            'Gamma': gamma,
            'Mean Accuracy': mean_accuracy,
            'Standard Deviation': std_deviation
        })

#print results
svc_optimization_df = pd.DataFrame(svc_optimization_results)
print("\nSVC Optimization Results:")
print(svc_optimization_df)




# print column names
print(rf_optimization_df.columns)
print(svc_optimization_df.columns)
print(knn_optimization_df.columns)





"""
This Section create the plots to show how all of the parameters compare in terms of mean accuracy
when the hyper parameters are adjusted on each classifier.

"""
#knn plot optimised results
plt.figure(figsize=(8, 6))


#adds error bars
plt.errorbar(knn_optimization_df['K'], knn_optimization_df['Mean Accuracy'], 
             yerr=knn_optimization_df['Standard Deviation'], fmt='o', color='b', ecolor='r', capsize=5)
# Add labels and title and plot
plt.xlabel("K Value")
plt.ylabel("Mean Accuracy")
plt.title('K-NN Optimization - K Value')
plt.grid(True)
plt.show()



# plot Random Forest optimisation results
plt.figure(figsize=(10, 6))

n_estimators_values = rf_optimization_df['n_estimators'].unique()

for n_estimators_value in n_estimators_values:
    rf_df = rf_optimization_df[rf_optimization_df['n_estimators'] == n_estimators_value]
    plt.errorbar(rf_df['max_depth'], rf_df['Mean Accuracy'],
                 yerr=rf_df['Standard Deviation'], fmt='-x', label=f'n_estimators = {n_estimators_value}',
                 capsize=5, linewidth=2, markersize=6, alpha=0.7)

# Add labels and title and plot
plt.title("Random Forest Optimization - Max Depth vs Accuracy", fontsize=14)
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.legend(title='Number of Estimators', loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Create a plot to compare RBF and Poly kernels
rbf_results = svc_optimization_df[svc_optimization_df['Kernel'] == 'rbf']
poly_results = svc_optimization_df[svc_optimization_df['Kernel'] == 'poly']


plt.figure(figsize=(8, 6))


# Plot RBF 
plt.plot(rbf_results['Gamma'], rbf_results['Mean Accuracy'], marker='x', label='RBF Kernel', color='blue', linestyle='-', linewidth=1,alpha = 0.5)

# Plot Poly 
plt.plot(poly_results['Gamma'], poly_results['Mean Accuracy'], marker='x', label='Poly Kernel', color='red', linestyle='-', linewidth=1, alpha= 0.5)

# Add labels and title and plot
plt.title('Comparison of RBF vs Poly Kernel for SVC', fontsize=14)
plt.xlabel('Gamma Value', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()




#All of the Optimised parameter values for each classifier
best_knn_k = 2
best_knn_mean_accuracy = 0.989521
best_knn_std_dev = 0.006359


best_rf_n_estimators = 200
best_rf_max_depth = 15
best_rf_mean_accuracy = 0.979702
best_rf_std_dev = 0.003803

best_svc_kernel = 'rbf'
best_svc_gamma = 0.001
best_svc_mean_accuracy = 0.990185
best_svc_std_dev = 0.004607


# re run the models using the best configurations
best_knn = KNeighborsClassifier(n_neighbors=best_knn_k, metric='minkowski')
best_rf = RandomForestClassifier(n_estimators=best_rf_n_estimators, max_depth=best_rf_max_depth, random_state=38)
best_svc = SVC(kernel=best_svc_kernel,  gamma=best_svc_gamma, random_state=38)


# retrain each model on the training data
best_knn.fit(X_train, y_train)
best_rf.fit(X_train, y_train)
best_svc.fit(X_train, y_train)

# predict on the test set
knn_pred = best_knn.predict(X_test)
rf_pred = best_rf.predict(X_test)
svc_pred = best_svc.predict(X_test)


# display the accuracy on the test set for each model
knn_test_accuracy = accuracy_score(y_test, knn_pred)
rf_test_accuracy = accuracy_score(y_test, rf_pred)
svc_test_accuracy = accuracy_score(y_test, svc_pred)

print(f"KNN Test Accuracy: {knn_test_accuracy}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy}")
print(f"SVC Test Accuracy: {svc_test_accuracy}")




#confusion matrix function
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    #Title and format of plot
    plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    
    # tick labels for x and y axes
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    
    #Label the matrix with counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# call cf function
plot_confusion_matrix(y_test, knn_pred, "K-NN")
plot_confusion_matrix(y_test, rf_pred, "RF")


plot_confusion_matrix(y_test, svc_pred, "SVC")





#plot the final values in a graph

#Plot accuracy and standard deviation of 3 models
optimized_accuracies = [best_rf_mean_accuracy, best_knn_mean_accuracy, best_svc_mean_accuracy]
std_devs = [best_rf_std_dev, best_knn_std_dev, best_svc_std_dev]

classifiers = ['Random Forest', 'KNN', 'SVC']

# Plotting the optimized mean accuracy with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(classifiers, optimized_accuracies, yerr=std_devs, fmt='o', 
             ecolor='red', capsize=5, color='blue', marker='s', markersize=8, label='Optimized Mean Accuracy')

# add labels and title
plt.xlabel('Classifiers')
plt.ylabel('Mean Accuracy')
plt.title('Optimized Mean Accuracy for Each Classifier with Standard Deviation')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
