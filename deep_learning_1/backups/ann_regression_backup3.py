from __future__ import print_function, division
# Note: you may need to update your version of future
# sudo pip install -U future
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# NOTE: some people using the default Python
# installation on Mac have had trouble with Axes3D
# Switching to Python 3 (brew install python3) or
# using Linux are both viable work-arounds
from lib.ann import ann

# generate the data
N = 500
X = np.random.random((N, 2))*4 - 2 # in between (-2, +2)
Y = X[:,0]*X[:,1] # 3D saddle
# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# make a neural network and train it
D = 2
M = 100 # number of hidden units
epochs = 200
ann_relu = ann_1h('regression', size_i=X.shape[1], size_h=100, activation='relu')
# TODO: add a test set to appreciate the impact of regularization.
history_loss, _ = ann_relu.fit(X, Y, learning_rate=1e-4, reg_rate=0.002, epochs=20000, verbose=True)
Yhat = ann_relu.predict(X)

# plot the costs
plt.plot(history_loss)
plt.show()

# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = ann_relu.predict(Xgrid)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()

# plot magnitude of residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
plt.show()


