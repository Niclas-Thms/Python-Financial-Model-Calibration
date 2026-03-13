import numpy as np
import math
from nelder_mead import * 
import matplotlib.pyplot as plt 


class PriceModel:
    def __init__(self):
        self.params_ = None        # (mu, omega, alpha, beta, nu)
        self.x_hat_ = None       
        self.loglik_ = None

    def loglik(self, r, x, sigma2_0=None):
        mu, omega, alpha, beta, nu = PriceModelUtils.unpack_params(x)
        n = len(r)
        eps = r - mu
        sigma2 = max(float(np.var(r, ddof=1)), 1e-12) if sigma2_0 is None else sigma2_0
        ll = 0.0
        for t in range(n):
            sigma = math.sqrt(sigma2)
            u = eps[t] / sigma
            ll += -math.log(sigma) + PriceModelUtils.student_t_logpdf_standardized(u, nu)
            sigma2 = omega + alpha * eps[t]**2 + beta * sigma2
        return -ll

    def fit(self, r):
        r = np.asarray(r, dtype=float)
        simplex0 = PriceModelUtils.get_initial_simplex(r)
        x_hat, fmin = nelder_mead(lambda x: self.loglik(r, x),simplex0)
        self.x_hat_ = x_hat
        self.params_ = PriceModelUtils.unpack_params(x_hat)
        self.loglik_ = -fmin
        return self.params_

    def simulate_prices(self, S0, timeSteps, sigma0):
        mu, omega, alpha, beta, nu = self.params_

        # drift risk-neutral par pas
        mu = 0.0 #0.002 / timeSteps 

        r = np.zeros(timeSteps)
        sigma2 = np.zeros(timeSteps)
        S = np.zeros(timeSteps)

        sigma2[0] = sigma0**2
        S[0] = S0

        for t in range(1, timeSteps):
            z = np.random.standard_t(nu) / np.sqrt(nu / (nu - 2))            
            r[t] = mu + np.sqrt(sigma2[t-1]) * z
            sigma2[t] = omega + alpha * (r[t-1] - mu)**2 + beta * sigma2[t-1]
            S[t] = S[t-1] * np.exp(r[t])

        return S, r, sigma2

        

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class PriceModelUtils:
    @staticmethod
    def unpack_params(x):
        omega = float(np.exp(x[1]))                       # > 0
        alpha = float(sigmoid(x[2]))                      # (0,1)
        beta = float((1.0 - alpha) * sigmoid(x[3]))       # alpha+beta < 1
        nu = float(2.0 + np.exp(x[4]))                    # > 2
        return x[0], omega, alpha, beta, nu # mu, omega, alpha, beta, nu
    
    @staticmethod
    def student_t_logpdf_standardized(u, nu):
        c = (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(math.pi * (nu - 2.0)))
        return c - 0.5 * (nu + 1.0) * np.log1p((u * u) / (nu - 2.0))
    
    @staticmethod
    def build_simplex(x0, steps):
        d = len(x0)
        simplex = np.zeros((d + 1, d))
        simplex[0] = x0.copy()
        for i in range(d):
            x = x0.copy()
            x[i] += steps[i]
            simplex[i + 1] = x
        return simplex
    
    @staticmethod
    def get_initial_simplex(r):
        mu0 = float(np.mean(r))
        var_r = float(np.var(r, ddof=1))

        omega0, alpha0, beta0, nu0 = 0.1 * var_r, 0.05, 0.70, 7.0

        w0 = math.log(omega0)
        a0 = math.log(alpha0 / (1 - alpha0))
        sb = beta0 / (1 - alpha0)
        b0 = math.log(sb / (1 - sb))
        v0 = math.log(nu0 - 2.0)

        x0 = np.array([mu0, w0, a0, b0, v0])
        steps = np.array([0.1*np.std(r),0.5,0.5,0.5,0.3])
        simplex0 = PriceModelUtils.build_simplex(x0, steps)
        return simplex0