# Compare accuracy of multiclass classification performed with:
# 1) a softmax output layer as the multiclass extension of the logistic regression
# 2) one hidden layer with different activation functions (plus the softmax output layer)
# python2 syntax (>> source venv2/bin/activate)

import numpy as np
from sklearn.utils import shuffle
from lib.ann import ann
from lib.process import get_data

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
cls_set = set(Y)
print str(D) + " features and " + str(len(cls_set)) + " target classes."
ann_dummy1 = ann(classification_set=cls_set, n_features=D, n_hidden_layers=0) #softmax only, no hidden layer
ann_dummy2 = ann(classification_set=cls_set, n_features=D, hidden_layers_size=5, activation='identity') #should be equivalent to no hidden layer
ann_sigmoid = ann(classification_set=cls_set, n_features=D, hidden_layers_size=5) #sigmoid hidden layer (default)
ann_tanh = ann(classification_set=cls_set, n_features=D, hidden_layers_size=5, activation='tanh')
ann_relu = ann(classification_set=cls_set, n_features=D, hidden_layers_size=5, activation='relu') #useful mainly with multilayer neural networks to reduce vanishing gradient issue

ann_dummy1_loss, ann_dummy1_acc, _, _ = ann_dummy1.fit(Xtrain, Ytrain)
ann_dummy2_loss, ann_dummy2_acc, _, _  = ann_dummy2.fit(Xtrain, Ytrain)
ann_sigmoid_loss, ann_sigmoid_acc, _, _  = ann_sigmoid.fit(Xtrain, Ytrain)
ann_tanh_loss, ann_tanh_acc, _, _  = ann_tanh.fit(Xtrain, Ytrain)
ann_relu_loss, ann_relu_acc, _, _  = ann_relu.fit(Xtrain, Ytrain)

print "Accuracy on Xtest with ann_dummy1:", np.mean(ann_dummy1.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_dummy2:", np.mean(ann_dummy2.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_sigmoid:", np.mean(ann_sigmoid.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_tanh:", np.mean(ann_tanh.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_relu:", np.mean(ann_relu.predict(Xtest) == Ytest)