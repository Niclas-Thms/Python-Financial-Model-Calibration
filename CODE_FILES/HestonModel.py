import numpy as np
from cmath import exp, log, sqrt # pour les complex
from scipy.integrate import quad_vec
from ImpliedVols import VanillaOption



class HestonModel:
    def __init__(self, spot, variance, rate, dividend, kappa, theta, sigma, rho):
        self._spot = spot
        self._variance = variance
        self._rate = rate
        self._dividend = dividend
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma
        self._rho = rho

    def Phi(self, u, t):
        i = 1j
        d = sqrt((self._rho * self._sigma * i * u - self._kappa) * (self._rho * self._sigma * i * u - self._kappa)
                       + self._sigma * self._sigma * (i * u + u * u))
        g = (self._kappa - self._rho * self._sigma * i * u - d) / (self._kappa - self._rho * self._sigma * i * u + d)

        return exp(
            i * u * (np.log(self._spot) + (self._rate - self._dividend) * t)
            + self._theta * self._kappa / (self._sigma * self._sigma)
            * ((self._kappa - self._rho * self._sigma * i * u - d) * t
            - 2 * log((1 - g * exp(-d * t)) / (1 - g))) + self._variance / (self._sigma * self._sigma)
            * (self._kappa - self._rho * self._sigma * i * u - d) * (1 - exp(-d * t)) / (1 - g * exp(-d * t)))

    def Price(self, option:VanillaOption):
        i = 1j
        maturity, strike = option.maturity, option.strike
        forward = self._spot * np.exp(self._rate * maturity)

        def f(u):
            v = complex(u, 0)
            f1 = (np.exp(-i * u * np.log(strike)) * self.Phi(v - i, maturity) / (i * u * forward)).real
            f2 = (np.exp(-i * u * np.log(strike)) * self.Phi(v, maturity) / (i * u)).real
            return forward * f1 - strike * f2
        
        integral = quad_vec(f, 0, 1000)[0] # no need to go to infinity
        price = np.exp(-self._rate * maturity) * (0.5 * (forward - strike) + integral / np.pi)
        if option.omega == -1:
            price = price - self._spot * np.exp(-self._dividend * maturity) + strike * np.exp(-self._rate * maturity)
        return price

    def simulate_antithetic(self, T, n_steps, n_paths, seed=10):
        S0, v0 = self._spot, self._variance
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        rng = np.random.default_rng(seed)
        Z1 = rng.standard_normal((n_paths, n_steps))
        Z2 = rng.standard_normal((n_paths, n_steps))

        dW1_p = sqrt_dt * Z2
        dW2_p = sqrt_dt * (self._rho * Z2 + np.sqrt(1 - self._rho**2) * Z1)
        dW1_m, dW2_m = -dW1_p, -dW2_p

        S_p, S_m = np.zeros((n_paths, n_steps + 1)), np.zeros((n_paths, n_steps + 1))
        v_p, v_m = np.zeros((n_paths, n_steps + 1)), np.zeros((n_paths, n_steps + 1))

        S_p[:, 0], S_m[:, 0] = S0, S0
        v_p[:, 0], v_m[:, 0] = v0, v0

        for i in range(1, n_steps + 1):
            sqrt_vp = np.sqrt(v_p[:, i-1])
            sqrt_vm = np.sqrt(v_m[:, i-1])

            S_p[:, i] = S_p[:, i-1] * (1.0 + (self._rate - self._dividend) * dt + sqrt_vp * dW2_p[:, i-1])
            S_m[:, i] = S_m[:, i-1] * (1.0 + (self._rate - self._dividend) * dt + sqrt_vm * dW2_m[:, i-1])

            v_prev_p = np.maximum(v_p[:, i-1], 0.0)
            v_prev_m = np.maximum(v_m[:, i-1], 0.0)
            v_p[:, i] = (v_p[:, i-1] + self._kappa * (self._theta - v_prev_p) * dt + self._sigma * np.sqrt(v_prev_p) * dW1_p[:, i-1])
            v_m[:, i] = (v_m[:, i-1] + self._kappa * (self._theta - v_prev_m) * dt + self._sigma * np.sqrt(v_prev_m) * dW1_m[:, i-1])

            # Numerical safety
            v_p[:, i], v_m[:, i] = np.maximum(v_p[:, i], 0), np.maximum(v_m[:, i], 0)

        return S_p, S_m