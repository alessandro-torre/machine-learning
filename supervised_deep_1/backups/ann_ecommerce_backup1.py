# Compare accuracy of multiclass classification performed with:
# 1) a softmax output layer as the multiclass extension of the logistic regression
# 2) an ANN with one hidden layer
# python2 syntax (>> source venv2/bin/activate)

import numpy as np
from sklearn.utils import shuffle
from lib.ann.backup1 import ann_1h
from lib.process import get_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
cls_set = set(Y)
ann0 = ann_1h(D, 0, cls_set) #softmax only, no hidden layer
ann3 = ann_1h(D, 3, cls_set) #proper ann

history0_c, history0_a = ann0.fit(Xtrain, Ytrain)
history3_c, history3_a = ann3.fit(Xtrain, Ytrain)

print "Accuracy on Xtest with ann0:", ann0.accuracy(ann0.predict(Xtest), Ytest)
print "Accuracy on Xtest with ann3:", ann3.accuracy(ann3.predict(Xtest), Ytest)
