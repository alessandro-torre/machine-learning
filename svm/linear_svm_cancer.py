import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from lib.svm import LinearSVC

try:
  import matplotlib.pyplot as plt
except ImportError:
  import matplotlib
  matplotlib.use('qt5agg')
  import matplotlib.pyplot as plt


# Get data
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Without features normalization
model = LinearSVC(normalize=False)

t0 = datetime.now()
losses = model.fit(X_train, Y_train)
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model.support_)}")
print(f"Coefficients w:\n{model.w}")
print(f"Intercept b: {model.b}")

plt.plot(losses)
plt.show()

print()

# With features normalization
model = LinearSVC(normalize=True)

t0 = datetime.now()
losses = model.fit(X_train, Y_train)
print(f"Training time: {datetime.now() - t0}")
print(f"Train accuracy: {model.score(X_train, Y_train):.2f}")
print(f"Test  accuracy: {model.score(X_test, Y_test):.2f}")
print(f"Number of support vectors: {len(model.support_)}")
print(f"Coefficients w:\n{model.w}")
print(f"Intercept b: {model.b}")

plt.plot(losses)
plt.show()
