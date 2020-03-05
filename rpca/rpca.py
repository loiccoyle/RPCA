import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.linalg import qr
from numpy.linalg import norm
import matplotlib.pyplot as plt

from .util import wthresh


class RobustPCA:
    def __init__(self, n_components=None, max_iter=100, tol=1e-5, beta=None, beta_init=None, gamma=0.5, mu=[5, 5], trim=False, verbose=True, copy=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.beta_init = beta_init
        self.gamma = gamma
        self.mu = mu
        self.trim = trim
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _initialisation(self, X): 

        n_samples, n_features = X.shape
        if self.beta is None:
            beta = 1/(2 * np.power(n_samples*n_features, 1/4));
        else:
            beta = self.beta
        if self.beta_init is None:
            beta_init = 4 * beta
        else:
            beta_init = self.beta_init
        if self.n_components is None:
            n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        zeta = beta_init * svds(X, k=1, return_singular_vectors=False)[0]
        S = wthresh(X, zeta)
        U, Sigma, V = svds(X - S, n_components)
        # transpose the V for consistency with matlab
        V = V.T
        # make Sigma a diag for consistency with matlab implementation
        Sigma = np.diag(Sigma)
        L = U @ Sigma @ V.T
        zeta = beta * Sigma[0, 0]
        S = wthresh(X - L ,zeta)

        self.beta_ = beta
        self.beta_init_ = beta_init
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self.n_components_ = n_components
        return L, S, U, Sigma, V

    def _fit(self, X):
        if self.copy:
            X = np.copy(X)

        errors = []
        norm_of_X = norm(X, 'fro')


        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        L, S, U, Sigma, V = self._initialisation(X)
        errors.append(norm(X - L - S, 'fro') / norm_of_X)

        for i in range(1, self.max_iter + 1):
            if self.trim:
                U, V = self._trim(U, Sigma[:self.n_components_, :self.n_components_], V, self.mu[0], self.mu[-1])
            # update L
            Z = X - S
            # print('Z', Z.shape)
            # print('V', V.shape)
            # print(Z @ V)
            # These 2 QR can be computed in parallel
            Q1, R1 = qr(Z.T @ U - V @ ((Z @ V).T @ U), mode='economic')
            Q2, R2 = qr(Z @ V - U @ (U.T @ Z  @ V), mode='economic')
            # print('R1', R1.shape)
            # print('R2', R2.shape)
            # print('upper_left', (U.T @ Z @ V).shape)

            M = np.vstack([np.hstack([U.T @ Z @ V, R1.T]),
                           np.hstack([R2, np.zeros_like(R2)])])
            # print('M', M.shape)
            U_of_M, Sigma, V_of_M = svd(M, full_matrices=False)
            V_of_M = V_of_M.T
            Sigma = np.diag(Sigma)
            # These 2 matrices multiplications can be computed in parallel
            U = np.hstack([U, Q2]) @ U_of_M[:, :self.n_components_]
            V = np.hstack([V, Q1]) @ V_of_M[:, :self.n_components_]
            L = U @ Sigma[:self.n_components_, :self.n_components_] @ V.T

            # update S
            zeta = self.beta_ * (Sigma[self.n_components_, self.n_components_] + ((self.gamma**i) * Sigma[0, 0]))
            S = wthresh(X - L, zeta)

            errors.append(norm(X - L - S, 'fro') / norm_of_X)

            if self.verbose:
                print(f'[{i}] Tolerance: {self.tol}\tCurrent error: {errors[i]}')
            if errors[i] < self.tol:
                print('Tolerance condition met.')
                break
        else:
            print('Tolerance condition not met.')
        self.L_ = L
        self.S_ = S
        self.U_ = U
        self.V_ = V
        self.Sigma_ = Sigma

        self.low_rank_ = L
        self.sparse_ = S
        # transpose V for consistency with sklearn's pca
        self.components_ = V.T
        # flatten the Sigma for  consistency with sklearn's pca
        self.singular_values_ = np.diag(Sigma)[:self.n_components_]

        self.end_iter_ = i
        self.errors_ = errors
        return L, S, U, Sigma, V

    def transform(self, X):
        if self.copy:
            X = np.copy(X)
        if self.mean_ is not None:
            X -= self.mean_
        return X @ self.components_.T

    def inverse_transform(self, X):
        return (X @ self.components_) + self.mean_  

    def fit_transform(self, X, y=None):
        _, _, U, Sigma, _ = self._fit(X)
        U = U[:, :self.n_components_]
        U *= np.diag(Sigma)[:self.n_components_]
        return U

    @staticmethod
    def __trim(X, mu_X):
        m, r = X.shape
        row_norm_square_X = np.sum(np.power(X, 2), axis=1)  # might need to set it to columns vector
        big_rows_X = row_norm_square_X > (mu_X * r / m)
        X[big_rows_X] = X[big_rows_X] * ((mu_X * r / m) / np.sqrt(row_norm_square_X[big_rows_X]))[:, np.newaxis]
        Q, R = qr(X, mode='economic')
        return Q, R

    def _trim(self, U, Sig, V , mu_V, mu_U):
        # these 2 qr can be computed in parallel
        Q1, R1 = self.__trim(U, mu_U)
        Q2, R2 = self.__trim(V, mu_V)
        U_tmp, _, V_tmp = svd(R1 @ Sig @ R2.T, full_matrices=False)
        return Q1 @ U_tmp, Q2 @ V_tmp.T