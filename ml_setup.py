# ML imports
from ucimlrepo import fetch_ucirepo 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fetch dataset 
# https://archive.ics.uci.edu/dataset/19/car+evaluation
car_evaluation = fetch_ucirepo(id=19) 
x = (car_evaluation.data.features).copy()
y = (car_evaluation.data.targets).copy()
# encoding
x = pd.get_dummies(x)
y = pd.get_dummies(y)

# splitting data into training and testing populations
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) # 70% training and 30% testing
clf = DecisionTreeClassifier(random_state=13, max_depth=8)
clf = clf.fit(x_train, y_train)
# predict the response for test dataset
y_pred = clf.predict(x_test)
y_probs = np.array(clf.predict_proba(x_test))[:, 1]

# cross-validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, x, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

def plot_decision_tree(clf_param):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, fontsize=10, filled=True)
    plt.tight_layout()
    plt.show()

def plot_roc_graph(y_test, y_probs):
    false_pos, true_pos, thresholds = roc_curve(y_test, y_probs)
    plt.plot(false_pos, true_pos)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

#plot_decision_tree(clf)
plot_roc_graph(y_test, y_probs)