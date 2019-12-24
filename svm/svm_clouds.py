import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from lib.svm import SVC

try:
  import matplotlib.pyplot as plt
except ImportError:
  import matplotlib
  matplotlib.use('qt5agg')
  import matplotlib.pyplot as plt


# Create two 2d gaussian clouds centered around [0,0] and [1,-1].
# Their centers are 6-sigma away [sigma = sqrt(2) / 6].
# Therefore, the clouds overlap with non-zero probability.
N = 1000
Y = np.array([0, 1] * (N//2))
X = list(zip(Y, -Y)) + np.random.randn(N, 2) * np.sqrt(2) / 6
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.3, cmap='seismic')
plt.title("Gaussian clouds")
plt.show()


# Train without momentum
model1 = SVC('linear')

# We save the initial weights to reuse with the model below
model1._build(X_train.shape)
alpha_init = model1.alpha.copy()
model1._built = True

t0 = datetime.now()
losses1 = model1.fit(X_train, Y_train, momentum=0)
print()
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model1.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model1.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model1.support_)}")


# Train with momentum
model2 = SVC('linear')

# For a proper comparison, we use the same initial weights
model2._build(X_train.shape)
model2.alpha = alpha_init.copy()
model2._built = True

t0 = datetime.now()
losses2 = model2.fit(X_train, Y_train, momentum=0.90)
print()
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model2.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model2.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model2.support_)}")


plt.plot(losses1, label="w/o momentum")
plt.plot(losses2, label="w/  momentum")
plt.legend()
plt.title("Loss")
plt.show()

model1.plot_decision_boundary(X_train, Y_train, title="w/o momentum")
model2.plot_decision_boundary(X_train, Y_train, title="w/  momentum")
