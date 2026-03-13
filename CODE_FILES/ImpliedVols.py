from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
from enum import Enum
from NewtonRaphson import NewtonRaphson 


class OptionType(Enum):
    Call = 'call'
    Put = 'put'

class Option:
    def __init__(self, strike, maturity, option_type: OptionType):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.option_type = option_type
        self.omega = 1.0 if option_type == OptionType.Call else -1.0

    def payoff(self, paths: np.ndarray):
        raise NotImplementedError("payoff must be implemented in subclasses")    

class VanillaOption(Option):
    def __init__(self, strike, maturity, option_type:OptionType):
        super().__init__(strike, maturity, option_type)
    
    def payoff(self, path):
        path = np.asarray(path)
        ST = path[-1] if path.ndim == 1 else path[:, -1]
        return np.maximum(self.omega * (ST - self.strike), 0.0)

class BlackScholesModel:
    def __init__(self, r, sigma=None):
        self.r = float(r)
        self.sigma = float(sigma) if sigma is not None else None

    def PriceCF(self, option:VanillaOption, S, vol=None):
        K, T, r, omega = option.strike, option.maturity, self.r, option.omega
        sigma = self.sigma if vol is None else vol
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return omega * (S * norm.cdf(omega * d1) - K * exp(-r * T) * norm.cdf(omega * (d1 - sigma * sqrt(T))))
    
    def Vega(self, option:VanillaOption, S, sigma=None):
        vol = self.sigma if sigma is None else sigma
        K, T, r = option.strike, option.maturity, self.r
        d1 = (log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * sqrt(T))
        return S * norm.pdf(d1) * sqrt(T)
    
    def solve_volatility(self, option:VanillaOption, S, target_price, vol_init=0.2):
        def f(vol):
            return self.PriceCF(option, S, vol) - target_price
        def jacobian(vol):
            return self.Vega(option, S, vol)
        nr = NewtonRaphson(f, vol_init, jacobian=jacobian)
        return nr.solve()[0]
                                       
