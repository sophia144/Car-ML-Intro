# ML imports
from ucimlrepo import fetch_ucirepo 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics

import matplotlib.pyplot as plt

# fetch dataset 
# https://archive.ics.uci.edu/dataset/19/car+evaluation
car_evaluation = fetch_ucirepo(id=19) 
x = (car_evaluation.data.features).copy()
y = (car_evaluation.data.targets).copy()

# replacing strings with ints
feature_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
x.replace(
    {'buying' : { 'vhigh' : 1, 'high' : 2, 'med' : 3, 'low' : 4 }, 
    'maint': { 'vhigh' : 1, 'high' : 2, 'med' : 3, 'low' : 4 }, 
    'doors' : {'2': 2, '3' : 3, '4' : 4, '5more': 5},
    'persons' : {'2' : 2, '4' : 4, 'more': 5},
    'lug_boot': { 'small' : 1, 'med' : 2, 'big' : 3 }, 
    'safety': { 'low' : 1, 'med' : 2, 'high' : 3 }},
    inplace = True
)

y.replace(
    {'class' : { 'unacc' : 1, 'acc' : 2, 'good' : 3, 'vgood' : 4 }},
    inplace = True
)


# splitting data into training and testing populations
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) # 70% training and 30% testing
clf = DecisionTreeClassifier(random_state=13, max_depth=8)
clf = clf.fit(x_train, y_train)
# predict the response for test dataset
y_pred = clf.predict(x_test)

#cross-validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, x, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# how often is the classifier correct
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, fontsize=10, filled=True)
plt.tight_layout()
#plt.show()