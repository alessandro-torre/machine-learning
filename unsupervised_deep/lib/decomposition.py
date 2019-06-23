import numpy as np

class PCA(object):
    def __init__(self):
        self.var = None  # eigenvalues
        self.Q = None    # matrix of eigenvectors

    def train(self, X):
        assert self.var is None, 'PCA already trained.'
        # Eigendecomposition of the covariance matrix
        var, Q = np.linalg.eigh(np.cov(X, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        # Order features from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        Q = Q[:, idx]
        self.var = var
        self.Q = Q

    def approximate(self, X, n_features=None):
        Z_ = self.transform(X, n_features)
        Q_ = self.Q[:,:n_features]
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        return Z_.dot(Q_.T)


    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'The PCA was not trained on the same number of features as X.'
        # Rotate the data to diagonalize its covariance matrix
        Z = X.dot(self.Q)
        return Z[:,:n_features]

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)


class SVD(object):
    def __init__(self):
        self.var = None  # eigenvalues
        self.S = None  # matrix of squared-root eigenvalues
        self.U = None  # matrix of eigenvectors of X.dot(X.T)
        self.V = None  # matrix of eigenvectors of X.T.dot(X)

    def train(self, X):
        assert self.var is None, 'SVD already trained.'
        # Eigendecomposition of the covariance matrix
        var, V = np.linalg.eigh(X.T.dot(X))
        # Eigendecomposition of X.dot(X.T). This naive implementation must be too slow!
        # Can we use np.linalg.eig() to solve X.T.dot(U) == X.inverse.dot(U).dot(lambda) ?
        var_, U = np.linalg.eigh(X.dot(X.T))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        var_ = np.maximum(var_, 0)
        # Order features and V from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        V = V[:, idx]
        # Order features and U from most to least significant
        # Not rigorous, but it should give U the same ordering as for V
        idx_ = np.argsort(-var_)
        var_ = var_[idx_]
        U = U[:, idx_]
        # Store results
        self.var = var
        self.S = np.diag(np.sqrt(var))
        self.U = U
        self.V = V

    def approximate(self, X, n_features=None):
        Z_ = self.transform(X, n_features)
        V_ = self.V[:,:n_features]
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        return Z_.dot(V_.T)

    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'The SVD was not trained on the same number of features as X.'
        # Truncate U and S
        U_ = self.U[:,:n_features]
        S_ = self.S[:n_features, :n_features]
        # Return data of reduced features
        # Note that X_ = U_.dot(S_).dot(V_.T) is a rank-k approximation of X,
        # but the *reduced* data is given by Z_ = X_.dot(V_) = U_.dot(S_),
        # which is analogous to Z = X.dot(Q) for PCA.
        # In this sense, X_ = Z_.dot(V_.T) is then a map back to the initial space.
        return U_.dot(S_)

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)
