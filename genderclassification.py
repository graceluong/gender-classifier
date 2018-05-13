from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

#[height (cm), weight(kg), shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39],
	[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 
	'female', 'male', 'female', 'male']

# initialize classifiers
clf_t = tree.DecisionTreeClassifier()
clf_s = SVC()

# train models
clf_t.fit(X,Y)
clf_s.fit(X,Y)

# test predictions
pred_t = clf_t.predict(X)
accu_t = accuracy_score(Y, pred_t)*100
print('The Accuracy for DecisionTree: {}'.format(accu_t))

pred_s = clf_s.predict(X)
accu_s = accuracy_score(Y, pred_s)*100
print('The Accuracy for SVC: {}'.format(accu_s))

# make prediction
prediction_t = clf_t.predict([[160, 50, 38]])
prediction_s = clf_s.predict([[160, 50, 38]])
print('DecisionTree says the measurements are from a {}'.format(prediction_t))
print('SVC says the measurements are from a {}'.format(prediction_s))

# best of the 2 classifiers
index = np.argmax([accu_t, accu_s])
classifiers = {0: 'DecisionTree', 1: 'SVC'}
print('Best gender classifier is {}'.format(classifiers[index]))