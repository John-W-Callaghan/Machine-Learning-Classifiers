# Description: 
# This script implements a Decision Tree Classifier to breast cancer data

# Created : 12th november 2024

# --------------------------------------------------
# Library dependencies:
# numpy, pandas, sklearn, matplotlib
#
# Input: Dataset containing breast cancer attributes.
# Output: 'Part2-decision-tree-display.png'
#         'Part2-ensemble.csv' - Predictions from the ensemble of decision trees.






import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



# load data
column_names = ['Class','age', 'menopause', 'tumor_size', 'inv_nodes','node-caps' ,'deg_malig','breast', 'breast_quad', 'irradiat']

# convert to dataframe
df = pd.read_csv('breast-cancer.data', names=column_names)


# remove nan + ? values from data
df.replace("?", np.nan, inplace=True)

df.dropna(inplace=True)


# attributes to encode
columns_to_encode = ['age', 'menopause', 'tumor_size', 'inv_nodes', 'node-caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']

# ordinal encoder method to encode data
ordinal_encoder = OrdinalEncoder()
df[columns_to_encode] = ordinal_encoder.fit_transform(df[columns_to_encode])

# label encode the Class variable
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])




# remove target y from dataframe
X = df.drop(columns=['Class'])
y = df['Class']

#Calculate values for summary
num_cases = df.shape[0]

num_attributes = X.shape[1]

num_classes = y.nunique()

class_distribution = y.value_counts().to_dict()  

# print dataset summary to console
print("# --------------------------------------------------")
print("# Dataset summary")
print(f"# Number of cases: {num_cases}")
print(f"# Number of attributes: {num_attributes}")
print(f"# Number of classes: {num_classes}")
print(f"# Class distribution (cases per class): {class_distribution}")
print("# List attributes and the number of different values")

# lists the attributes and unique values
for i, col in enumerate(X.columns):
    unique_values = X[col].nunique()
    print(f"# Attribute[{i}]: {col} ({unique_values} unique values)")



# Split data intotraining and test 80/20 (use stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=38)



# Initialize the Decision tree classifier
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=38)

decision_tree.fit(X_train, y_train)

#predict on both training and test data
y_train_pred = decision_tree.predict(X_train)
y_test_pred = decision_tree.predict(X_test)

# calculate the accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy in correct format
print("# --------------------------------------------------")
print("# DecisionTreeClassifier")
print(f"# criterion: {decision_tree.criterion}")
print(f"# max_depth: {decision_tree.max_depth}")
print(f"# random_state: {decision_tree.random_state}")
print(f"# accuracy_score(on training set): {train_accuracy:.2f}")
print(f"# accuracy_score(on test set): {test_accuracy:.2f}")
print("# --------------------------------------------------")



# Convert feature names to a list
feature_names = list(X.columns)

# plot the decision tree and save it as an image file
plt.figure(figsize=(16, 10))
plot_tree(decision_tree, feature_names=feature_names, class_names=['No Recurrence', 'Recurrence'], filled=True)
plt.title("Decision Tree Classifier with max_depth=5")
plt.savefig("Part2-decision-tree-display.png")  
plt.show()


# 2c. -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


SEED = 38

classifiers = []

# Train nine decision trees with different max_depth values 1-9
for depth in range(1, 10):  
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, max_features=9, random_state=SEED)
    clf.fit(X_train, y_train)
    classifiers.append(clf)
    print(f"Trained Decision Tree {depth} with max_depth={depth}")


all_predictions = []




#Loop through each case and store predictions for the ith case
for i in range(X_test.shape[0]):
    case_predictions = []  
    
    for clf in classifiers:
        # Get the prediction from each classifier for the i-th test case
        case_predictions.append(clf.predict(X_test.iloc[i:i+1])[0])  # gets the first element of the array
    
    # Store the predictions for this test case
    all_predictions.append(case_predictions)




final_predictions = []


def majority_voting(predictions):
    """
    takes a list of classifier predictions, this function returns the majority vote.
    
    Parameters:
    - predictions NP.ARRAY
    
    Returns:
    - int: The majority vote class label.
    """
    predictions = np.array(predictions)  #Convert predictions to a numpy array
    
    # Count the class labels and and how many time it appears
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    # Find the class with the highest count
    majority_class = unique_classes[np.argmax(counts)]
    
    return majority_class


# Loop through each test case's predictions
for case_predictions in all_predictions:

    # Apply majority voting to get the final prediction for this case
    ensemble_prediction = majority_voting(case_predictions)
    final_predictions.append(ensemble_prediction)





#Create a list to store the predictions
results = []

for idx, (case_predictions, actual_class, ensemble_prediction) in enumerate(zip(all_predictions, y_test, final_predictions), start=1):
    results.append({
        'case_id': idx,
        **{f'dt{i+1}': case_predictions[i] for i in range(9)},  # Predictions from dt1 to dt9
        'actual_class': actual_class,
        'ensemble': ensemble_prediction
    })

# Save predictions to dataframe then create csv
results_df = pd.DataFrame(results)
results_df.to_csv("Part2-ensemble.csv", index=False)

# Compute accuracy on the test set
ensemble_accuracy = accuracy_score(y_test, final_predictions)
print("# --------------------------------------------------")
print("Ensemble")
print(f"# Ensemble accuracy_score(on test set): {ensemble_accuracy:.2f}")


