from typing import Optional, Tuple

import numpy.typing as npt
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.linalg import qr
from numpy.linalg import norm

from .util import wthresh


class NotFittedError(Exception):
    pass


class RobustPCA:
    """Robust Principal Component Analylsis using Accelerated Alternating Projections.

    This class implements the Robust Principal Component Analysis (RPCA) algorithm using
    Accelerated Alternating Projections as described in [1]_.

    This implementation is a python port of the matlab implementation by HanQin Cai, et. al. available at:
        https://github.com/caesarcai/AccAltProj_for_RPCA

    It follows the scikit-learn API.

    .. [1] HanQin Cai, et. al. Accelerated alternating projections for robust principal component analysis.
        (https://arxiv.org/abs/1711.05519)

    Args:
        n_components: The number of low-rank components to extract. If None, it is automatically set.
        max_iter: The maximum number of iterations to perform.
        tol: The relative tolerance for convergence.
        beta: The regularization parameter for the sparse component. If None, it is automatically set.
        beta_init: The initial value for the regularization parameter for the sparse component. If None, it is
            automatically set.
        gamma: The step size for the ADMM algorithm.
        mu: The ADMM penalty parameters for the low-rank and sparse components.
        trim: Whether to trim small values in the sparse component to zero.
        verbose: Whether to print progress messages during fitting.
        copy: Whether to make a copy of the input data matrix before fitting.

    Examples:
        Fit a RobustPCA estimator to the input data matrix X, and reduce the dimension of X.

        >>> import numpy as np
        >>> from rpca import RobustPCA
        >>> X = np.random.rand(100, 50)
        >>> rpca = RobustPCA(n_components=10)
        >>> rpca.fit(X)
        >>> X_pca = rpca.transform(X)
        >>> X_pca.shape
        (100, 10)

    Raises:
        ValueError: If the number of low-rank components is not positive, or if the maximum number of iterations
                    is not positive.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        beta: Optional[float] = None,
        beta_init: Optional[float] = None,
        gamma: float = 0.5,
        mu: Tuple[float, float] = (5, 5),
        trim: bool = False,
        verbose: bool = True,
        copy: bool = True,
    ):
        if n_components is not None and n_components <= 0:
            raise ValueError(
                f"Expected positive number of components, got {n_components} instead."
            )
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

    def fit(self, X: npt.ArrayLike, y=None) -> "RobustPCA":
        """Fit the estimator to the input data.

        Args:
            X: An array-like matrix of shape (n_samples, n_features) representing the input data to fit.
            y: Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns:
            The fitted estimator object.

        Raises:
            ValueError: If the input data matrix is not a 2D array.
        """
        self._fit(np.asarray(X))
        return self

    def _initialisation(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        # svds requires float dtype
        X = np.asarray(X, dtype=float)

        n_samples, n_features = X.shape
        if self.beta is None:
            beta = 1 / (2 * np.power(n_samples * n_features, 1 / 4))
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

        zeta: float
        zeta = beta_init * svds(X, k=1, return_singular_vectors=False)[0]  # type: ignore
        S = wthresh(X, zeta)

        U: npt.NDArray
        Sigma: npt.NDArray
        V: npt.NDArray
        U, Sigma, V = svds(X - S, n_components)  # type: ignore
        # make Sigma a diag for consistency with matlab implementation
        Sigma = np.diag(Sigma)
        L = U @ Sigma @ V
        zeta = beta * Sigma[0, 0]
        S = wthresh(X - L, zeta)

        self.beta_ = beta
        self.beta_init_ = beta_init
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self.n_components_ = n_components
        # transpose the V for consistency with matlab
        return L, S, U, Sigma, V.T

    def _fit(
        self, X: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        if np.ndim(X) != 2:
            raise ValueError(
                f"Expected 2D array, got {np.ndim(X)}D array instead:\n{X}"
            )

        if self.copy:
            X = np.copy(X)

        errors = []
        norm_of_X: float
        norm_of_X = norm(X, "fro")  # type: ignore

        self.mean_ = np.mean(X, axis=0)
        X = np.subtract(X, self.mean_)

        L, S, U, Sigma, V = self._initialisation(X)
        errors.append(self._compute_error(X, L, S, norm_of_X))

        i = 1
        for i in range(1, self.max_iter + 1):
            if self.trim:
                U, V = self._trim(
                    U,
                    Sigma[: self.n_components_, : self.n_components_],
                    V,
                    self.mu[0],
                    self.mu[-1],
                )
            # update L
            Z = X - S
            # These 2 QR can be computed in parallel
            Q1: npt.NDArray
            R1: npt.NDArray
            Q2: npt.NDArray
            R2: npt.NDArray
            Q1, R1 = qr(Z.T @ U - V @ ((Z @ V).T @ U), mode="economic")  # type: ignore
            Q2, R2 = qr(Z @ V - U @ (U.T @ Z @ V), mode="economic")  # type: ignore

            M = np.vstack(
                [np.hstack([U.T @ Z @ V, R1.T]), np.hstack([R2, np.zeros_like(R2)])]
            )
            U_of_M, Sigma, V_of_M = svd(M, full_matrices=False)
            V_of_M = V_of_M.T
            Sigma = np.diag(Sigma)
            # These 2 matrices multiplications can be computed in parallel
            U = np.hstack([U, Q2]) @ U_of_M[:, : self.n_components_]
            V = np.hstack([V, Q1]) @ V_of_M[:, : self.n_components_]
            L = U @ Sigma[: self.n_components_, : self.n_components_] @ V.T

            # update S
            zeta = self.beta_ * (
                Sigma[self.n_components_, self.n_components_]
                + ((self.gamma**i) * Sigma[0, 0])
            )
            S = wthresh(X - L, zeta)

            error = self._compute_error(X, L, S, norm_of_X)
            errors.append(error)

            if self.verbose:
                print(f"[{i}] Tolerance: {self.tol}\tCurrent error: {error}")
            if error < self.tol:
                print("Tolerance condition met.")
                break
        else:
            print("Tolerance condition not met.")
        self.L_ = L
        self.S_ = S
        self.U_ = U
        self.V_ = V
        self.Sigma_ = Sigma

        self.low_rank_ = L
        self.sparse_ = S
        # transpose V for consistency with sklearn's pca
        self.components_ = V.T
        # flatten the Sigma for consistency with sklearn's pca
        self.singular_values_ = np.diag(Sigma)[: self.n_components_]

        self.end_iter_ = i
        self.errors_ = errors
        return L, S, U, Sigma, V

    def transform(self, X: npt.ArrayLike) -> npt.NDArray:
        """Apply dimensionality reduction to X.

        Args:
            X: An array-like matrix of shape (n_samples, n_features) to transform.

        Returns:
            An ndarray of shape (n_samples, n_features) representing the transformed data matrix, with the low-rank and sparse components removed.

        Raises:
            NotFittedError: If the estimator has not been fitted to any data.
        """
        if not hasattr(self, "components_"):
            raise NotFittedError(
                "This RobustPCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        if self.copy:
            X = np.copy(X)
        if self.mean_ is not None:
            X = np.subtract(X, self.mean_)
        return X @ self.components_.T

    def inverse_transform(self, X: npt.ArrayLike) -> npt.NDArray:
        """Transform data back to its original space.

        Args:
            X: An array-like matrix of shape (n_samples, n_features) representing the transformed data matrix to inverse transform.

        Returns:
            An ndarray of shape (n_samples, n_features) representing the inverse transformed data matrix, with the low-rank and sparse components added back.

        Raises:
            NotFittedError: If the estimator has not been fitted to any data.
        """
        if not hasattr(self, "components_"):
            raise NotFittedError(
                "This RobustPCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        return (X @ self.components_) + self.mean_

    def fit_transform(self, X: npt.ArrayLike, y=None) -> npt.NDArray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X: An array-like matrix of shape (n_samples, n_features) representing the input data to fit and transform.
            y: Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns:
            An ndarray of shape (n_samples, n_features) representing the transformed data matrix,
            with the low-rank and sparse components removed.

        Raises:
            ValueError: If the input data matrix is not a 2D array.
        """
        _, _, U, Sigma, _ = self._fit(np.asarray(X))
        U = U[:, : self.n_components_]
        U *= np.diag(Sigma)[: self.n_components_]
        return U

    @staticmethod
    def __trim(X: npt.NDArray, mu_X: float) -> Tuple[npt.NDArray, npt.NDArray]:
        m, r = X.shape
        row_norm_square_X = np.sum(
            np.power(X, 2), axis=1
        )  # might need to set it to columns vector
        big_rows_X = row_norm_square_X > (mu_X * r / m)
        X[big_rows_X] = (
            X[big_rows_X]
            * ((mu_X * r / m) / np.sqrt(row_norm_square_X[big_rows_X]))[:, np.newaxis]
        )
        Q: npt.NDArray
        R: npt.NDArray
        Q, R = qr(X, mode="economic")  # type: ignore
        return Q, R

    def _trim(
        self, U: npt.NDArray, Sig: npt.NDArray, V: npt.NDArray, mu_V: float, mu_U: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        # these 2 qr can be computed in parallel
        Q1, R1 = self.__trim(U, mu_U)
        Q2, R2 = self.__trim(V, mu_V)
        U_tmp, _, V_tmp = svd(R1 @ Sig @ R2.T, full_matrices=False)
        return Q1 @ U_tmp, Q2 @ V_tmp.T

    @staticmethod
    def _compute_error(
        X: npt.NDArray, L: npt.NDArray, S: npt.NDArray, norm_of_X: Optional[float]
    ) -> float:
        return norm(X - (L + S), "fro") / (
            norm(X, "fro") if norm_of_X is None else norm_of_X
        )  # type:ignore
