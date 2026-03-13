import numpy as np
from ImpliedVols import *
from HestonModel import *





def generate_uniform_population(n_particles, bounds):
    bounds = np.array(bounds, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    dim = len(bounds)
    particles = []
    while len(particles) < n_particles:
        x = np.random.uniform(low=low, high=high, size=dim)
        v0, kappa, theta, rho, sigma = x 
        if 2.0 * kappa * theta > sigma * sigma:
            particles.append(x)
    return particles


LAMBDA_3 = 1e3
LAMBDA_5 = 1e5
THETA = 0.16276168110077988**2

class HestonEstimator:
    def __init__(self, data_dict, S0, r):
        self.data = data_dict
        self.S0 = S0
        self.r = r
        self.bs = BlackScholesModel(r)
        self.maturities = list(data_dict.keys())
        self.options = {}       # VanillaOption
        self.forwards = {}
        self.n = len(self.maturities) * len(data_dict[self.maturities[0]]["strikes"])
        self._precompute_market_quantities()

    def _precompute_market_quantities(self):
        for T in self.maturities:
            d = self.data[T]
            strikes = d["strikes"]
            F = d["Forward"]
            self.forwards[T] = F
            self.options[T] = []
            for K in strikes:
                opt = VanillaOption(K, T, OptionType.Call)
                self.options[T].append(opt)

    def objective(self, x):
        v0, kappa, theta_bar, rho, sigma = x
        if v0 <= 0 or theta_bar <= 0 or sigma <= 0 or kappa <= 0 or kappa > 50:
            return np.inf
        if not (-0.999999 < rho < 0.999999) or 2.0 * kappa * theta_bar <= sigma * sigma:
            return np.inf

        model = HestonModel(self.S0, v0, self.r, 0.0, kappa, theta_bar, sigma, rho)
        error, penalty = 0.0, 0.0
        anchor_strikes = {98, 102} # strikes to enforce “smile stability across maturities”
        prev_vol = {}
        for T in self.maturities:
            for opt, p_mkt in zip(self.options[T], self.data[T]["call"]):
                price = model.Price(opt)
                error += (price - p_mkt) ** 2

                # Question 5: penalize variation of model implied vol across maturities
                if len(self.maturities) > 1 and opt.strike in anchor_strikes:
                    vol = self.bs.solve_volatility(opt, self.S0, price)
                    if opt.strike in prev_vol:
                        penalty += LAMBDA_5 * (vol - prev_vol[opt.strike]) ** 2
                    prev_vol[opt.strike] = vol

        if len(self.maturities) > 1:
            return error / self.n + penalty
        return error / self.n + LAMBDA_3 * (theta_bar - THETA)**2






class HestonCalibrator:

    def __init__(self, data_dict, S0, r):
        self.S0 = S0
        self.r = r
        self.strikes = [99, 100, 101]
        strike_to_idx = {K: i for i, K in enumerate(data_dict[1.0]["strikes"])}

        self.options = [VanillaOption(K, 1.0, OptionType.Call) for K in self.strikes]   # VanillaOption
        calls = np.asarray(data_dict[1.0]["call"])
        self.market_prices = calls[[strike_to_idx[K] for K in self.strikes]]        
        self.forward = S0 * exp(r)

    def objectiveExact(self, theta):
        theta_bar, kappa = THETA, 0.05
        v0, rho, sigma = theta if len(theta) != 1 else theta[0]
        if np.abs(rho) > 1 or v0 <= 0 or kappa <= 0 or theta_bar <= 0 or sigma <= 0:
            return [np.inf, np.inf, np.inf]
        model = HestonModel(spot=self.S0,variance=v0,rate=self.r,dividend=0.0,kappa=kappa,theta=theta_bar,sigma=sigma,rho=rho)
        return [model.Price(self.options[i]) - self.market_prices[i] for i in range(len(self.strikes))]
    
    def objective(self, theta):
        theta_bar, kappa = THETA, 0.05
        v0, rho, sigma = theta
        error = 0.0
        if v0 <= 0 or kappa <= 0 or theta_bar <= 0 or sigma <= 0:
            return np.inf
        if abs(rho) >= 1 :#or 2 * kappa * theta_bar <= sigma**2: # correl
            return np.inf
        model = HestonModel(spot=self.S0,variance=v0,rate=self.r,dividend=0.0,kappa=kappa,theta=theta_bar,sigma=sigma,rho=rho)
        for opt, mkt_price in zip(self.options,self.market_prices):
            price = model.Price(opt)
            error += (price - mkt_price)**2
        return error / len(self.options)

