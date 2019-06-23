import numpy as np
'''
This library contains two simple classes to perform PCA and SVD on given data X.
X is assumed to come in the standard form of features in the columns, and one
data point per row.

PCA and SVD are similar, but SVD does PCA twice: on the right singular vectors
of X (as PCA does), and on the left singular vectors of X.
Notation used here:
    X: input data
    var: non-zero singular values of X, i.e. eigenvalues of X.T.dot(X)/2
    S: diagonal matrix with sqrt(2 * eigenvalues), such that S^2/2 = diag(var)
    V: matrix of right singular vectors of X, i.e. eigenvectors of X.T.dot(X)/2
    U: matrix of left singular vectors of X, i.e. eigenvectors of X.dot(X.T)/2
    Z = X.dot(V): rotated data, with covariance Z.T.dot(Z)/2 = diag(var)
PCA transforms X by reducing V:
    V_ = V[:,:n_features]
    Z_ = X.dot(V_)     [rotated and reduced]
    X_ = Z_.dot(V_.T)  [reconstructed]
SVD transforms X by reducing both U and V (and S as a consequence):
    U_ = U[:,:n_features]
    V_ = V[:,:n_features]
    S_ = S[:n_features, :n_features]
    Z_ = U_.dot(S_)    [rotated and reduced]
    X_ = Z_.dot(V_.T)  [reconstructed]
X_ = Z_.dot(V_.T) is simply a map back to the initial space.

Note that SVD "removes more" by reducing U, but the transformed data has the
same reduced variance matrix as with PCA. We can show this by using
U.dot(U.T) = V.dot(V.T) = 1 (which hold for U_ and V_ as well), S.T = S,
and the fact that V is the eigenvector matrix of X.T.dot(X)/2:
> PCA: Z_.T.dot(Z_)/2 = V_.T.dot(X.T).dot(X).dot(V_)/2
       = V_.T.dot(V_).dot(diag(var_)) = diag(var_)
> SVD: Z_.T.dot(Z_)/2 = S_.T.dot(U_.T).dot(U_).dot(U_)/2
       = S_.T.dot(S_)/2 = = (S_)^2/2 = diag(var_)
where var_ = var_[:n_features], i.e. the reduced vector of eigenvalues.
'''


class PCA(object):
    def __init__(self):
        self.var = None  # eigenvalues of cov(X.T)
        self.V = None    # matrix of eigenvectors of cov(X.T)

    def train(self, X):
        assert self.var is None, 'PCA already trained.'
        # Eigendecomposition of the covariance matrix
        var, V = np.linalg.eigh(np.cov(X, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        # Sort features from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        V = V[:, idx]
        # Store result
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
        self.var = None  # eigenvalues of cov(X.T)
        self.S = None  # matrix of squared-root eigenvalues of cov(X.T)
        self.Sinv = None  # inverse of S
        self.V = None  # matrix of eigenvectors of cov(X.T)

    def train(self, X, smoothing=1e-3):
        assert self.var is None, 'SVD already trained.'
        # Eigendecomposition of the covariance matrix
        var, V = np.linalg.eigh(np.cov(X, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision.
        var = np.maximum(var, 0)
        # Sort features and V from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        V = V[:, idx]
        # Store result
        self.var = var
        self.V = V
        # We add smoothing to be able to invert S, which we need in train.
        # Note also the sqrt(2) normalization.
        s = np.sqrt(2 * var + smoothing)
        self.S = np.diag(s)
        self.Sinv = np.diag(1/s)

    def approximate(self, X, n_features=None):
        Z_ = self.transform(X, n_features)
        V_ = self.V[:,:n_features]
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        return Z_.dot(V_.T)

    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'The SVD was not trained on the same number of features as X.'
        # Calculate U from X, V, S.
        # No need to solve the eigendecomposition of the NxN matrix X.dot(X.T),
        # which grows O(N^2), with N = X.shape[0] (number of input data points).
        U = X.dot(self.V).dot(self.Sinv)
        # Truncate U and S
        U_ = U[:,:n_features]
        S_ = self.S[:n_features, :n_features]
        # Return data of reduced features
        # Note that X_ = U_.dot(S_).dot(V_.T) is a rank-k approximation of X,
        # but the *reduced* data is given by Z_ = X_.dot(V_) = U_.dot(S_).
        return U_.dot(S_)

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)
