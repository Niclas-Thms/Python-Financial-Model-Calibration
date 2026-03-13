import numpy as np
from math import exp, sqrt
from HestonModel import HestonModel
from ImpliedVols import *
from enum import Enum
from scipy.stats import norm


class StdType(Enum):
    Antithetic = "anthi"
    Antithetic_Call = "anthi_cv_c"
    NONE = "none"

class AsianOption(Option):
    def __init__(self,strike: float,maturity: float,option_type: OptionType,include_S0: bool = True):
        super().__init__(strike, maturity, option_type)
        self.include_S0 = include_S0

    def payoff(self, paths: np.ndarray):
        if not self.include_S0:
            paths = paths[:, 1:]
        return np.maximum(self.omega * (paths.mean(axis=1) - self.strike), 0.0)
    
class HestonPricer:
    def __init__(self, model: HestonModel, n_steps, n_paths):
        self.model = model
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.Sp = None
        self.Sm = None

    def price(self, option: Option, method:StdType):
        if self.Sm is None and self.Sp is None:
            self.Sp, self.Sm = self.model.simulate_antithetic(option.maturity, self.n_steps, self.n_paths)
        disc = exp(-self.model._rate * option.maturity)
        X = 0.5 * disc * ((option.payoff(self.Sp) + option.payoff(self.Sm)) 
                          if method == StdType.Antithetic or method == StdType.Antithetic_Call
                          else 2 * option.payoff(self.Sp))
        if method == StdType.NONE or method == StdType.Antithetic:
            return {"price": X.mean(), "se": norm.ppf(0.995) * X.std(ddof=1) / sqrt(len(X))}
        else: # antithetic + variable de contrôle call heston
            expectation_Y = self.model.Price(VanillaOption(option.strike, option.maturity, OptionType.Call))
            control_option = VanillaOption(option.strike,option.maturity,option.option_type)
            Y = 0.5 * disc * (control_option.payoff(self.Sp) + control_option.payoff(self.Sm))
            beta = np.cov(X, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)
            Z = X - beta * (Y - expectation_Y) # E(Z)=E(X) but V(Z)<V(X)
            mean_diff = (Y - expectation_Y).mean()
            se_diff = norm.ppf(0.995) * (Y - expectation_Y).std(ddof=1) / sqrt(len(Y))
            return {"price": Z.mean(),"se": norm.ppf(0.995) * Z.std(ddof=1) / sqrt(len(Z)),
                     "IC_cv":[mean_diff - se_diff, mean_diff + se_diff]}