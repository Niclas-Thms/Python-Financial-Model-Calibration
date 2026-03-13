import numpy as np
from numpy.linalg import inv
from nelder_mead import *

def first_derivative(k, b, rho, m, sigma):
    return b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))

def second_derivative(k, b, m, sigma):
    return b * (sigma**2) / (((k-m)**2 + sigma**2)**(3/2))

def g(k, T, a, b, rho, m, sigma):
    w = svi_implied_vol(k, T, a, b, rho, m, sigma)**2 * T
    w_ = first_derivative(k, b, rho, m, sigma)
    w__ = second_derivative(k, b, m, sigma)
    return (1 - 0.5 * k * w_ / w)**2 - 0.25 * w_**2 * (1 / w + 0.25) + 0.5 * w__

def svi_implied_vol(k, T, a, b, rho, m, sigma):
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return np.sqrt(w / T)

class SVIRawCalibrator:
    def __init__(self, strikes, vols, vegas, forward, T):
        self.k = np.log(np.array(strikes) / forward)
        self.w = (np.array(vols)**2) * T
        self.vegas = np.array(vegas)
        self.T = T

    def _regress_abc(self, m, sigma, weights):
        y = (self.k - m) / sigma
        z = np.sqrt(1.0 + y*y)

        X = np.column_stack([np.ones_like(y), y, z])
        W = np.diag(weights)

        B = inv(X.T @ W @ X) @ X.T @ W @ self.w
        return B[0], B[2] / sigma, B[1] / B[2] # a, b, rho

    def objective(self, x):
        sigma, m = x
        if sigma <= 0 or m < self.k.min() or m > self.k.max():
            return np.inf

        weights = self.vegas**2
        a, b, rho = self._regress_abc(m, sigma, weights)
        if a < 0 or a > self.w.max() or abs(rho) >= 1:
            return np.inf
        if b < 0 or b > 4 / (1 + abs(rho)):
            return np.inf
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return np.inf

        vol_svi = svi_implied_vol(self.k, self.T, a, b, rho, m, sigma)
        err = weights * (vol_svi - np.sqrt(self.w / self.T))**2
        return np.mean(err)

    def calibrate(self):
        x0 = [[0.010, -0.010],[0.005, 0.005],[0.030, 0.030]]
        res = nelder_mead(self.objective, simplex=x0)
        sigma, m = res[0]
        a, b, rho = self._regress_abc(m, sigma, self.vegas)
        return [a,b,rho,m,sigma], self.objective(res[0])
    



    