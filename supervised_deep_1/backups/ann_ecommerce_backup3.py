# Compare accuracy of multiclass classification performed with:
# 1) a softmax output layer as the multiclass extension of the logistic regression
# 2) one hidden layer with different activation functions (plus the softmax output layer)
# python2 syntax (>> source venv2/bin/activate)

import numpy as np
from sklearn.utils import shuffle
from lib.ann import ann_1h
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
ann_dummy1 = ann_1h('classification', D, 0, cls_set) #softmax only, no hidden layer
ann_dummy2 = ann_1h('classification', D, 5, cls_set, activation='identity') #should be equivalent to no hidden layer
ann_sigmoid = ann_1h('classification', D, 5, cls_set) #sigmoid hidden layer
ann_tanh = ann_1h('classification', D, 5, cls_set, activation='tanh')
ann_relu = ann_1h('classification', D, 5, cls_set, activation='relu') #useful mainly with multilayer neural networks

ann_dummy1_loss, ann_dummy1_acc = ann_dummy1.fit(Xtrain, Ytrain)
ann_dummy2_loss, ann_dummy2_acc = ann_dummy2.fit(Xtrain, Ytrain)
ann_sigmoid_loss, ann_sigmoid_acc = ann_sigmoid.fit(Xtrain, Ytrain)
ann_tanh_loss, ann_tanh_acc = ann_tanh.fit(Xtrain, Ytrain)
ann_relu_loss, ann_relu_acc = ann_relu.fit(Xtrain, Ytrain)

print "Accuracy on Xtest with ann_dummy1:", np.mean(ann_dummy1.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_dummy2:", np.mean(ann_dummy2.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_sigmoid:", np.mean(ann_sigmoid.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_tanh:", np.mean(ann_tanh.predict(Xtest) == Ytest)
print "Accuracy on Xtest with ann_relu:", np.mean(ann_relu.predict(Xtest) == Ytest)