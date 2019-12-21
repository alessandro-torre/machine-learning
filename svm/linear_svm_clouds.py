import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from lib.svc import LinearSVC

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
plt.show()


model = LinearSVC()

t0 = datetime.now()
losses = model.fit(X_train, Y_train)
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model.support_)}")
print(f"Coefficients w:\n{model.w}")
print(f"Intercept b: {model.b:.2f}")

plt.plot(losses)
plt.show()

model.plot_decision_boundary(X_train, Y_train)
