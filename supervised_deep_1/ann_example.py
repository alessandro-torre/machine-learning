import numpy as np
from lib.ann import ann
#hack for matplotlib in venv on a mac
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('svg')
from matplotlib import pyplot as plt

def get_data(N_per_class=500):
	D = 2 #size of each input point
	K = 3 #number of classes
	N = N_per_class * K #input size
	X1 = np.random.randn(N_per_class, D) + np.array([0, -2])
	X2 = np.random.randn(N_per_class, D) + np.array([2, 2])
	X3 = np.random.randn(N_per_class, D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])
	Y = np.array(["A"]*N_per_class + ["B"]*N_per_class + ["C"]*N_per_class)
	assert X.shape[0] == Y.shape[0]
	return X, Y

def main():
	Xtrain, Ytrain = get_data(500)
	Xtest, Ytest = get_data(500)
	#define the ann
	D = Xtrain.shape[1] #size of each input point
	class_set = set(Ytrain)
	model = ann(classification_set=class_set, n_features=D, hidden_layers_size=3)
	#train
	history_c, history_a, _, _ = model.fit(Xtrain, Ytrain, verbose=True, adaptive_rate=True)
	plt.plot(history_c)
	plt.show()
	#plt.savefig('history_c.svg', bbox_inches='tight')
	#test
	print "Accuracy on test data:", np.mean(model.predict(Xtest) == Ytest)

#TODO: trace validation set along to train, and plot both histories

if __name__ == '__main__':
	main()
