import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from lib.svm import SVC
from lib.svm_kernels import Kernel

try:
  import matplotlib.pyplot as plt
except ImportError:
  import matplotlib
  matplotlib.use('qt5agg')
  import matplotlib.pyplot as plt


# Create four 2d gaussian clouds centered around [0,0], [0,1], [1,0] and [1,1].
# Labels are assigned according to xor rule applied to coordinates of center.
# Their centers are 6-sigma away [sigma = sqrt(2) / 6].
# Therefore, the clouds overlap with non-zero probability.
N = 500
c1 = np.array([0, 0, 1, 1] * N)  # center coordinate 1
c2 = np.array([0, 1, 0, 1] * N)  # center coordinate 2
Y  = np.array([0, 1, 1, 0] * N)  # xor of c1, c2
X = np.array(list(zip(c1, c2)) + np.random.randn(4*N, 2) * np.sqrt(2) / 6)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.3, cmap='seismic')
plt.title("Gaussian clouds")
plt.show()


# Polynomial kernel
model1 = SVC('polynomial')

t0 = datetime.now()
losses1 = model1.fit(X_train, Y_train, momentum=0.)
print()
print(f"{model1.name} kernel")
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model1.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model1.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model1.support_)}")


# Rbf kernel
model2 = SVC('rbf')

t0 = datetime.now()
losses2 = model2.fit(X_train, Y_train, momentum=0.90)
print()
print(f"{model2.name} kernel")
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model2.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model2.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model2.support_)}")


# Rbf kernel with lower gamma (less prone to overfitting)
model3 = SVC(Kernel('rbf', gamma=0.1))
model3.name = 'rbf (gamma=0.1)'

t0 = datetime.now()
losses3 = model3.fit(X_train, Y_train, momentum=0.90)
print()
print(f"{model3.name} kernel")
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model3.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model3.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model3.support_)}")


# Compare losses and decision boundaries
plt.plot(losses1, label=model1.name)
plt.plot(losses2, label=model2.name)
plt.plot(losses3, label=model3.name)
plt.legend()
plt.title("Loss")
plt.show()

model1.plot_decision_boundary(X_train, Y_train)
model2.plot_decision_boundary(X_train, Y_train)
model3.plot_decision_boundary(X_train, Y_train)
