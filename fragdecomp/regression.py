import numpy as np
from warnings import warn
import scipy
from numba import jit

import pymc3 as pm


class BayesianRegression(object):

    def __init__(self, X, y, y_err, prior_sd=500, informative_priors=True):
        """Class to handle the bayesian linear regression for y ~ beta * X.
        Assumes X does not have an intercept term included.

        Parameters
        ----------
        X : numpy.ndarray
            A (n, m) matrix of predictor variables
        y : np.ndarray
            A (n,) vector of responses
        y_err : np.ndarray
            A (n,) vector of estimated measurement variances for y (used as
            weights)
        prior_sd: float
            An estimate for the prior standard deviation in the regressor means.

        """

        X = np.asarray(X)
        y = np.asarray(y)

        self.n, self.m = n, m = X.shape
        assert len(y) == n, "Y vector shape mismatch"

        # Add an intercept term
        X_intercept = np.hstack([np.ones(n)[:, np.newaxis], X])
        if np.linalg.matrix_rank(X_intercept) < (m+1):
            warn("X Matrix is not full-rank")

        # Augment matrix to account for prior distribution. From Gelman, 14.24
        # (mean 0, std=500)
        y_star = np.hstack([y, np.zeros(m+1)])
        self._x_star = X_star = np.vstack([X_intercept, np.eye(m+1)])
        self._n_star = n_star = len(y_star)

        # Covariance matrix for the measurements
        Qy = np.eye(n_star)
        Qy[:n,:n] = np.diag(y_err)
        Qy[-(m+1):,-(m+1):] *= prior_sd**2
    
        # Compute the cholesky and inverse cholesky matrices
        self._qy_chol = Qy_chol = np.linalg.cholesky(Qy)
        Qy_inv = np.linalg.inv(Qy_chol)

        # Scale the explanatory and response matrices (Gelman 14.17 & 14.18)
        X_star_q = Qy_inv @ X_star
        y_star_q = Qy_inv @ y_star

        # Compute the QR factorization, find the MAP beta point
        q, r = scipy.linalg.qr(X_star_q, mode='economic', pivoting=False)
        self.beta_hat = scipy.linalg.solve_triangular(r, q.T @ y_star_q)

        # Following procedure from (14.5) and below.
        self._r_inv = scipy.linalg.lapack.dtrtri(r)[0]
        # vb = r_inv @ r_inv.T

        res = (y_star_q - X_star_q @ self.beta_hat)
        self._s_squared = (res @ res.T) / (n_star - (m+1))

    
    @jit
    def sample(self, num=200):
        """Sample the fitted regression. 

        Parameters
        ----------
        num : int
            Number of draws

        Returns
        -------
        sigma_squared : A (num,) array of estimated squared errors
        beta: A (num, m+1) array, with intercept and slope terms
        y_hat: A (num, n) arr

        """

    
        beta = np.zeros((num, self.m + 1))
        y_hat = np.zeros((num, self.n))
        sigma_squared = (
            ((self._n_star - (self.m + 1)) * self._s_squared) /
            np.random.chisquare( (self._n_star - (self.m + 1)), size=num))

        for i in range(num):
            beta[i] = (self.beta_hat +
                       (self._r_inv * np.sqrt(sigma_squared[i])) @
                       np.random.randn(self.m + 1))
            y_hat[i] = (self._x_star @ beta[i] + 
                        (self._qy_chol * np.sqrt(sigma_squared[i])) @
                        np.random.randn(self._n_star))[:self.n]
            
        return sigma_squared, beta, y_hat


    def predict(self, X, beta, centered=True):
        """Predict the regressed mean and std for a set of explanatory variables and
        trace of slopes"""

        means = beta[:, 1:] @ X.T + np.atleast_2d(beta[:,0]).T
        median = np.median(means, 0)
        hpd = pm.hpd(means)

        if centered:
            return median, np.abs(hpd - median[:, np.newaxis])
        else:
            return median, hpd


class SVDBayesianRegression(BayesianRegression):

    def __init__(self, X, y, y_err, prior_sd=500):
        """A regression class which operates based on the singlar value
        decomposition of the regressor variables """

        u, s, v = np.linalg.svd(X, full_matrices=False)
        rank = (s > 1E-8).sum()

        self._u = u
        self._s = s
        self._v = v
        self._rank = rank

        BayesianRegression.__init__(self, u[:, :rank], y, y_err, prior_sd)

    
    def sample(self, num=200):

        sigma_squared, alpha, y_hat = BayesianRegression.sample(self, num)
        beta_no_slope = (np.linalg.pinv(
            np.diag(self._s)[:self._rank, :self._rank] 
            @ self._v[:self._rank]) @ alpha[:, 1:].T)
        beta = np.concatenate([np.atleast_2d(alpha[:,0]), beta_no_slope]).T

        return sigma_squared, beta, y_hat

    
    def is_outlier(self, X):

         return np.abs(self._v[self._rank:] @ X.T).sum(0) > 1E-10


class BayesianRegressionOutlier(SVDBayesianRegression, BayesianRegression):

    def __init__(self, X, y, y_err, prior_sd=500):
        """ A class which uses the BayesianRegression class for the base
        regression, while using the SVDBayesianRegression class for outlier
        detection.
        """

        BayesianRegression.__init__(self, X, y, y_err, prior_sd)

        u, s, v = np.linalg.svd(X, full_matrices=False)
        rank = (s > 1E-8).sum()

        self._u = u
        self._s = s
        self._v = v
        self._rank = rank


    def sample(self, num=200):
        return BayesianRegression.sample(self, num)
