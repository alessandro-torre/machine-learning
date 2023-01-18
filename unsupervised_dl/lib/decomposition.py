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

Note that SVD seems to "remove more" by reducing U, but SVD and PCA actually
return the same Z_. We can first show that the variance of Z_ is the same
by using
    U.dot(U.T) = V.dot(V.T) = 1 (which hold for U_ and V_ as well)
    S.T = S,
and the fact that V is the eigenvector matrix of X.T.dot(X)/2:
> PCA: Z_.T.dot(Z_)/2 = V_.T.dot(X.T).dot(X).dot(V_)/2
       = V_.T.dot(V_).dot(diag(var_)) = diag(var_)
> SVD: Z_.T.dot(Z_)/2 = S_.T.dot(U_.T).dot(U_).dot(U_)/2
       = S_.T.dot(S_)/2 = = (S_)^2/2 = diag(var_)
where var_ = var_[:n_features], i.e. the reduced vector of eigenvalues.

To show that Z_ is actually the same for PCA and SVD, we look at the SVD steps.
To calculate U, there is no need to solve the eigendecomposition of the ((N,N))
matrix X.dot(X.T). This would be unfeasable, since the operation grows O(N^2),
where N = X.shape[0] (number of input data points). Instead, we can use the
equality X = U.dot(S).dot(V.T). By reversing it:
    U  = X.dot(V).dot(Sinv) = Z.dot(Sinv)
We can see that U is a normalised version of Z, since we divide by sqrt(2*var):
    cov(U.T) = U.T.dot(U)/2 = diag(1/2)
What can we say about U_?
    U_ = U[:,:n_features]
If we want a reduced U_, it is actually faster to first *right* truncate Sinv
and then multiply, instead of truncating the resulting U:
    Sinv_ = Sinv[:,:n_features]  (right-truncated only!)
    U_ = X.dot(self.V).dot(Sinv_)
At this point, X_ = U_.dot(S_).dot(V_.T) is the rank-k approximation of X, but
the *reduced* data (analogous to PCA) is given by:
    Z_ = X_.dot(V_) = U_.dot(S_)
where
    S_ = S[:n_features, :n_features]
Now notice that:
    Sinv_.dot(S_) = diag(1_)
which right-truncate V to V_, when right multiplied to it:
    V.dot(Sinv_).dot(S_) = V.diag(1_) = V_
Now, if we were to return Z_, we would do:
    Z_ = U_.dot(S_) = X.dot(V).dot(Sinv_).dot(S_) = X.dot(V_)
We have found the same result of PCA.

Our implementation of SVD is the same as PCA. The only difference is that
SVD.transform() gives the option to normalize the transformed data.
If we want to return not normalized data, we return Z_. We don't need to
calculate Sinv_ and U_, since this would just add more intermediate steps.
If we want to return normalized data, we return sqrt(2)*U_. This is the
normalised version of Z_, since we essentially divide by sqrt(var):
    cov(sqrt(2)*U_.T) = U_.T.dot(U_) = diag(1_)
'''


class PCA(object):
    def __init__(self):
        self.var = None  # eigenvalues of cov(X.T)
        self.V = None    # matrix of eigenvectors of cov(X.T)

    def train(self, X):
        assert self.var is None, 'Model already trained.'
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
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        V_ = self.V[:,:n_features]
        Z_ = X.dot(V_)  # or Z_ = self.transform(X, n_features), but slower
        X_ = Z_.dot(V_.T)
        return X_

    def transform(self, X, n_features=None):
        assert X.shape[1]==len(self.var), 'Model not trained on the same number of features as X.'
        # Rotate the data to diagonalize its covariance matrix.
        # Note: it is faster to first truncate (V) and then multiply.
        V_ = self.V[:,:n_features]
        Z_ = X.dot(V_)
        return Z_

    def train_and_transform(self, X, n_features=None):
        self.train(X)
        return self.transform(X, n_features)


class SVD(object):
    def __init__(self):
        self.var = None  # eigenvalues of cov(X.T)
        self.V = None  # matrix of eigenvectors of cov(X.T)

    def train(self, X):
        assert self.var is None, 'Model already trained.'
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

    def approximate(self, X, n_features=None):
        # Reconstruct X_ with a mapping to the initial space.
        # This gives a rank-k approximation of X (here k=n_features).
        # Not that for SVD we should do:
        #    U_ = self.transform(X, n_features) / sqrt(2)
        #    S_ = S[:n_features, :n_features]
        #    V_ = V[:,:n_features]
        #    X_ = U_.dot(S_).dot(V_.T)
        # But we know (see the derivation above) that:
        #    U_.dot(S_) = X.dot(V).dot(Sinv_).dot(S_) = X.dot(V_)
        # so it is smarter (less steps) to just do the same as for PCA:
        V_ = self.V[:,:n_features]
        Z_ = X.dot(V_)
        X_ = Z_.dot(V_.T)
        return X_

    def transform(self, X, n_features=None, normalize=True, smoothing=1e-3):
        assert X.shape[1]==len(self.var), 'Model not trained on the same number of features as X.'
        # As described above, there is no need to solve the eigendecomposition
        # of X.dot(X.T), to find U. Instead, we use
        #    U = X.dot(V).dot(Sinv)
        # where
        #    Sinv = np.diag(1 / sqrt(2 * var))
        Z = X.dot(self.V)
        # If normalize is False, this is the same as PCA.transform()
        if normalize:
            # Note that we add smoothing to be able to invert var.
            # We also drop the sqrt(2) normalization of Sinv, since we would then
            # need to return sqrt(2) * Z.
            Sinv = np.diag(1 / np.sqrt(self.var + smoothing))
            Z = Z.dot(Sinv)  # this is sqrt(2)*U
        return Z[:,:n_features]

    def train_and_transform(self, X, n_features=None, normalize=True, smoothing=1e-3):
        self.train(X)
        return self.transform(X, n_features, smoothing)
