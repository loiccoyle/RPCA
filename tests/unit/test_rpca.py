import unittest
import numpy as np
from rpca import RobustPCA


class TestRPCA(unittest.TestCase):
    def test_initialisation(self):
        # Test case 1: Test with default parameters
        X = np.random.rand(100, 50)
        rpca = RobustPCA()
        L, S, U, Sigma, V = rpca._initialisation(X)
        self.assertEqual(L.shape, (100, 50))
        self.assertEqual(S.shape, (100, 50))
        self.assertEqual(U.shape, (100, rpca.n_components_))
        self.assertEqual(Sigma.shape, (rpca.n_components_, rpca.n_components_))
        self.assertEqual(V.shape, (50, rpca.n_components_))

        # Test case 2: Test with custom parameters
        X = np.random.rand(50, 100)
        rpca = RobustPCA(beta=0.1, beta_init=0.2, n_components=5)
        L, S, U, Sigma, V = rpca._initialisation(X)
        self.assertEqual(L.shape, (50, 100))
        self.assertEqual(S.shape, (50, 100))
        self.assertEqual(U.shape, (50, rpca.n_components_))
        self.assertEqual(Sigma.shape, (rpca.n_components_, rpca.n_components_))
        self.assertEqual(V.shape, (100, rpca.n_components_))

        # Test case 3: Test with custom X
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        rpca = RobustPCA()
        L, S, U, Sigma, V = rpca._initialisation(X)
        self.assertEqual(L.shape, (3, 3))
        self.assertEqual(S.shape, (3, 3))
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(Sigma.shape, (2, 2))
        self.assertEqual(V.shape, (3, 2))

    def test_fit(self):
        X = np.random.rand(100, 50)
        rpca = RobustPCA(n_components=10)
        rpca.fit(X)
        self.assertEqual(rpca.low_rank_.shape, (100, 50))
        self.assertEqual(rpca.sparse_.shape, (100, 50))

    def test_fit_transform(self):
        X = np.random.rand(100, 50)
        rpca = RobustPCA(n_components=2)
        X_pca = rpca.fit_transform(X)
        self.assertEqual(rpca.low_rank_.shape, (100, 50))
        self.assertEqual(rpca.sparse_.shape, (100, 50))
        self.assertEqual(X_pca.shape, (100, 2))
