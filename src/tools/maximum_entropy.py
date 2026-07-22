from tools.models import diffusion3D
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MaxEnt:

    def __init__(self, t, correlation, std=None):
        self.t = t
        self.y = correlation

        self.n_t = len(t)
 
        if std is not None:
            self.std = np.abs(std)
            self.std = np.array([std if std > 1e-3 else 1e-3 for std in self.std])
            self.std = np.array([std if std < 1e3 else 1e3 for std in self.std])
            self.std = self.std * (self.n_t / np.sum(self.std))
        else:
            self.std = np.ones_like(correlation)

        self.variance = self.std**2

    def set_fitting_space(self, tau: np.ndarray, alpha):
        self.tau = tau
        self.n_tau = len(tau)
        self.tauMax = np.max(tau)
        self.tauMin = np.min(tau)

        self.alpha = alpha

    def initial_guess(self, params=None):
        self.params = params
        self.T = self.fidelity_matrix(params)
        self.singular_value_decomposition()

        self.dist = np.ones(self.n_tau) / self.n_tau
        self.y_fit = self.evaluate(params)
        return np.copy(self.y_fit)

    def fidelity_matrix(self, params):
        # N.B. offset is not used, because the offset has already been subtracted from the data
        if params is None:
            self.offset = 0
        else:
            a, self.offset = params

        T = np.zeros((self.n_t, self.n_tau))
        for i_t, t in enumerate(self.t):
            for i_tau, tau in enumerate(self.tau):
                T[i_t, i_tau] = diffusion3D(t/tau, a)
        return T

    def singular_value_decomposition(self, f=0):
        U, s, VT = svd(self.T, full_matrices=True)
        self.n_s = np.sum(s >= (s[0] * f))
        self.S = np.diag(s[:self.n_s])
        self.U = U[:, :self.n_s]
        self.V = VT[:self.n_s].T
        return self.S, self.U, self.V

    def evaluate(self, params=None):
        params = params or self.params
        corr = self.fidelity_matrix(params) @ self.dist + self.offset
        return corr
    
    def predict(self, t):
        """Calculate the acf(t)"""
        a, _ = self.params
        T = np.array([diffusion3D(t/tau, a) for tau in self.tau])
        acf_value = T @ self.dist
        return acf_value

    def loss(self, params):
        y_fit = self.fidelity_matrix(params) @ self.dist
        y = self.y - params[5]
        fit_penalty = np.sum((y - y_fit)**2 / self.variance)
        return fit_penalty
    
    def fit(self, guess, cycles=10, n_params=5, n_dist=5, bounds=None, plot=False):
        if n_dist:
            self.fit_distribution(n_dist)

        acf_fit = self.y_fit
        y = 1
        return acf_fit, y
    
    def fit_params(self, p0, num_iter, bounds=None):
        self.params = minimize(self.loss, p0, method='Nelder-Mead', bounds=bounds, options={'maxiter': num_iter}).x

    def fit_distribution(self, num_iter):
        y = (self.y - self.offset)

        # Initial distribution
        w = np.ones(self.n_tau) / self.n_tau
        u = np.linalg.lstsq(self.V, np.log(self.dist/w))[0]

        # == Optimization (Bryan1990)
        I = np.eye(self.n_s)
        M = self.S @ self.U.T @ np.diag(1/self.variance) @ self.U @ self.S
        for _ in range(num_iter):
            # == Obtain A
            K = self.V.T @ np.diag(self.dist) @ self.V
            A = M @ K +  self.alpha*I  # fidelity + regularization (small values are preffered)

            # == Obtain b 
            self.y_fit = self.U @ self.S @ self.V.T @ self.dist  # Equiv. to T @ self.dist
            g = self.S.T @ self.U.T @ ((self.y_fit - y) / self.variance)
            b = -self.alpha*u - g

            # == Solve for du and update distribution
            du = np.linalg.lstsq(A, b, rcond=None)[0]
            u += du
            self.dist = w * np.exp(self.V @ u)
            
        # Final adjustments
        self.y_fit = self.U @ self.S @ self.V.T @ self.dist
        self.dist = np.where(np.isfinite(self.dist), self.dist, 0.0) 
        self.N = 1/np.sum(self.dist)