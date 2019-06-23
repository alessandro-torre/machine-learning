import numpy as np

class PCA(object):
    def __init__(self):
        self.var = None  # eigenvalues
        self.V = None    # matrix of eigenvectors

    def train(self, X):
        assert self.var is None, 'PCA already trained.'
        # Eigendecomposition of the covariance matrix
        var, V = np.linalg.eigh(np.cov(X, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        # Order features from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        V = V[:, idx]
        self.var = var
        self.V = V

    def approximate(self, X, n_features=None):
        Z_ = self.transform(X, n_features)
        V_ = self.V[:,:n_features]
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        return Z_.dot(V_.T)

    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'The PCA was not trained on the same number of features as X.'
        # Rotate the data to diagonalize its covariance matrix.
        # Note: it is faster to truncate before multiplying.
        V_ = self.V[:,:n_features]
        return X.dot(V_)

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)


class SVD(object):
    def __init__(self):
        self.var = None  # eigenvalues
        self.S = None  # matrix of squared-root eigenvalues
        self.Sinv = None  # inverse of S
        self.V = None  # matrix of eigenvectors

    def train(self, X, smoothing=1e-3):
        assert self.var is None, 'SVD already trained.'
        # Eigendecomposition of the covariance matrix
        var, V = np.linalg.eigh(np.cov(X, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision.
        # We add smoothing to be able to invert S, which we need in train.
        var = np.maximum(var, smoothing)
        # Order features and V from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        V = V[:, idx]
        # Store results
        self.var = var
        self.S = np.diag(np.sqrt(var))
        self.Sinv = np.diag(1/np.sqrt(var))
        self.V = V

    def approximate(self, X, n_features=None):
        Z_ = self.transform(X, n_features)
        V_ = self.V[:,:n_features]
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        return Z_.dot(V_.T)

    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'The SVD was not trained on the same number of features as X.'
        # Calculate U from X, V, S
        U = X.dot(self.V).dot(self.Sinv)
        # Truncate U and S
        U_ = U[:,:n_features]
        S_ = self.S[:n_features, :n_features]
        # Return data of reduced features
        # Note that X_ = U_.dot(S_).dot(V_.T) is a rank-k approximation of X,
        # but the *reduced* data is given by Z_ = X_.dot(V_) = U_.dot(S_),
        # which is analogous to Z_ = X.dot(V_) for PCA.
        # In this sense, X_ = Z_.dot(V_.T) is then a map back to the initial space.
        # Finally, PCA reduces U.dot(S), while SVD reduces both U and S, before multiplying.
        return U_.dot(S_)

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)
