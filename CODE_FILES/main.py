from math import exp
from ImpliedVols import *
from HestonModel import HestonModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#################################    1    ##############################################
S0, r = 100, 0.0020

data_dict = {0.5: {"Forward": S0 * exp(r * 0.5),
                   "strikes": [95,96,97,98,99,100,101,102,103,104],
                   "call": [], "put": [], "volatilities": []},
             1.0: {"Forward": S0 * exp(r * 1.0),
                   "strikes": [95,96,97,98,99,100,101,102,103,104],
                   "call": [10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46],
                   "put": [], "volatilities": []},
             1.5: {"Forward": S0 * exp(r * 1.5),
                   "strikes": [95,96,97,98,99,100,101,102,103,104],
                   "call": [], "put": [], "volatilities": []}}

model = BlackScholesModel(r)
T = 1.0
d = data_dict[T]
for i, K in enumerate(d["strikes"]):
    call_price = d["call"][i]
    option = VanillaOption(K, T, OptionType.Call)
    vol = model.solve_volatility(option, S0, call_price)
    d["volatilities"].append(vol)
    d["put"].append(call_price - S0 + K * exp(-r * T))

base_vols = d["volatilities"]
for T in [0.5, 1.5]:
    d = data_dict[T]
    strikes = d["strikes"]
    d["volatilities"] = base_vols.copy()
    for i, K in enumerate(strikes):
        option = VanillaOption(K, T, OptionType.Call)
        call = model.PriceCF(option, S0, base_vols[i])
        d["call"].append(call)
        d["put"].append(call - S0 + K * exp(-r * T))


#################################     2    ##############################################
def VIX(maturity: float):
    data = data_dict[maturity]
    F, calls, puts, strikes = data["Forward"], data["call"], data["put"], data["strikes"]
    K0 = max(K for K in strikes if K <= F)
    sum_prices, n = 0.0, len(strikes)

    for i, K in enumerate(strikes):
        if i == 0:
            deltaK = strikes[i+1] - strikes[i]
        elif i == n - 1:
            deltaK = strikes[i] - strikes[i-1]
        else:
            deltaK = 0.5 * (strikes[i+1] - strikes[i-1])

        Q = puts[i] if K < F else calls[i] # pas de K=F, on peut donc avoir condition K >= F
        sum_prices += deltaK * Q / (K * K)
    variance = (2 * exp(r*maturity) * sum_prices - (F / K0 - 1) ** 2) / maturity
    return 100 * sqrt(variance)












simplex_1_low_vol_near_flat = [
    [0.040, 1.20, 0.040, -0.10, 0.25],
    [0.055, 1.20, 0.040, -0.10, 0.25],
    [0.040, 1.55, 0.040, -0.10, 0.25],
    [0.040, 1.20, 0.060, -0.10, 0.25],
    [0.040, 1.20, 0.040,  0.10, 0.25],
    [0.040, 1.20, 0.040, -0.10, 0.35],
]
simplex_2_high_mean_reversion = [
    [0.030, 4.50, 0.030, -0.40, 0.45],
    [0.045, 4.50, 0.030, -0.40, 0.45],
    [0.030, 5.00, 0.030, -0.40, 0.45],
    [0.030, 4.50, 0.045, -0.40, 0.45],
    [0.030, 4.50, 0.030,  0.20, 0.45],
    [0.030, 4.50, 0.030, -0.40, 0.60],
]
simplex_3_positive_correlation_region = [
    [0.050, 2.00, 0.050,  0.50, 0.35],
    [0.070, 2.00, 0.050,  0.50, 0.35],
    [0.050, 2.60, 0.050,  0.50, 0.35],
    [0.050, 2.00, 0.080,  0.50, 0.35],
    [0.050, 2.00, 0.050, -0.80, 0.35],
    [0.050, 2.00, 0.050,  0.50, 0.55],
]
simplex_4_high_sigma_but_feasible = [
    [0.100, 3.00, 0.120, -0.70, 0.80],
    [0.130, 3.00, 0.120, -0.70, 0.80],
    [0.100, 3.90, 0.120, -0.70, 0.80],
    [0.100, 3.00, 0.160, -0.70, 0.80],
    [0.100, 3.00, 0.120,  0.10, 0.80],
    [0.100, 3.00, 0.120, -0.70, 0.95],
]
simplex_5_high_theta_long_run_var = [
    [0.120, 0.80, 0.300, -0.20, 0.55],
    [0.160, 0.80, 0.300, -0.20, 0.55],
    [0.120, 1.10, 0.300, -0.20, 0.55],
    [0.120, 0.80, 0.380, -0.20, 0.55],
    [0.120, 0.80, 0.300,  0.40, 0.55],
    [0.120, 0.80, 0.300, -0.20, 0.75],
]
simplexes = [
    simplex_1_low_vol_near_flat,
    simplex_2_high_mean_reversion,
    simplex_3_positive_correlation_region,
    simplex_4_high_sigma_but_feasible,
    simplex_5_high_theta_long_run_var,
]



from nelder_mead import nelder_mead
from PSO import *
import time
from HestonCalib import *


#################################    3 & 5    ##############################################
hmodel = HestonModel(S0, 0.04, 0.025, 0.0, 1.5, 0.04, 0.3, -0.9)
prices = [hmodel.Price(VanillaOption(S0, T, OptionType.Call)) for T in range(1,16)]
print(prices) # check prices with the article

def multi_calibration(estimator,bounds,n_runs=5):
    results = []
    for run in range(n_runs):
        simplex0 = simplexes[run]
        pop = generate_uniform_population(100, bounds)
        start = time.time()
        #x_best, f_best = nelder_mead(objective=estimator.objective,simplex=simplex0,max_iter=200)
        x_best, f_best = particle_swarm(estimator.objective, pop)
        t = time.time() - start
        results.append((x_best, f_best))
        print(f"[Run {run+1:02d}] time {t} f = {f_best:.6e} x = {x_best}")
    results_sorted = sorted(results, key=lambda x: x[1])
    return results_sorted

bounds = [
    (0.0001, 0.5),     # v0
    (0.0001, 5.0),      # kappa
    (0.01, 0.5),     # theta
    (-0.99, 0.99),    # rho
    (0.05, 3.0)      # sigma
]

d ={1.0:data_dict[1.0]} # question 3
#d = data_dict          # question 5
estimator = HestonEstimator(d, S0=100, r=0.002)
#res = multi_calibration(estimator, bounds)


############################### Question 4 ##################################################
calibrator = HestonCalibrator(d, 100, 0.002)
# pop = generate_uniform_population(50, bounds)
# p = [[p[1], p[3], p[4]] for p in pop] # get v0, rho, sigma
# x0, f0 = particle_swarm(calibrator.objective, p)
# print(x0, calibrator.objectiveExact(x0))
# nr = NewtonRaphson(calibrator.objectiveExact, x0, tol=1e-2)
# res = nr.solve()
# print(res)
# print(calibrator.objectiveExact(res))



##################################    6    ##############################################
from PricerMC import *

# heston_params = [("Q3", 0.0453, 2.18, 0.0265, 0.34, 0.32),("Q4", 0.2858, 0.05, 0.0265, 8.08, 0.81),
#                  ("Q5", 0.0321, 0.15, 0.0379, 0.11, 0.50),]
# option = AsianOption(strike=98, maturity=1.0, option_type=OptionType.Call)
# methods = [(StdType.NONE, "Asian option price"),(StdType.Antithetic, "Asian option price (antithetic)"),
#     (StdType.Antithetic_Call, "Asian option price (antithetic + control variate)")]

# def print_result(label, res):
#     ic_low  = res["price"] - res["se"]
#     ic_high = res["price"] + res["se"]
#     msg = (f"{label}: {res['price']:.6f}, Standard error: {res['se']:.6f}, IC(99%)=[{ic_low:.6f}, {ic_high:.6f}]")
#     if "IC_cv" in res:
#         msg += f", IC control variate={res['IC_cv']}"
#     print(msg)

# for name, v0, kappa, theta, sigma, rho in heston_params:
#     model = HestonModel(S0, v0, r, 0.0, kappa, theta, sigma, rho)
#     pricer = HestonPricer(model, n_steps=252, n_paths=100_000)
#     for method, label in methods:
#         res = pricer.price(option, method=method)
#         print_result(label, res)











from TimeSeriesStats import TimeSeriesStats

def log_returns(prices):
    prices = np.array(prices)
    return np.log(prices[1:] / prices[:-1])

data = pd.read_excel("Historical dataset TD 5-6.xlsx")
returns = log_returns(data.iloc[:, 1])

jb = TimeSeriesStats.jarque_bera(returns)
print(f"JB stat = {jb}")
#TimeSeriesStats.plot_acf(returns, nlags=20)
#TimeSeriesStats.plot_acf(returns**2, nlags=20)

from PriceModel import PriceModel
from PriceModelDiagnostic import *

pm = PriceModel()
pm.fit(returns)
params = pm.params_

print(params)
#Diagnostics.run(returns, pm, nlags=30)



from WeightedMC import WeightedMC




import numpy as np

def simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    return ST

def build_call_payoffs(ST, strikes, r, T):
    disc = np.exp(-r*T)
    G = np.zeros((len(ST), len(strikes)))
    for j, K in enumerate(strikes):
        G[:, j] = disc * np.maximum(ST - K, 0.0)
    return G

bs = BlackScholesModel(0.0020)
S0, r, sigma, T = 100.0, 0.002, 0.20, 1.0

strikes = np.array(data_dict[1.0]["strikes"], float)
C_mkt = np.array(data_dict[1.0]["call"], float)
C_mkt_bs = [bs.PriceCF(VanillaOption(K, 1.0, OptionType.Call), S0, 0.2) for K in strikes]

# ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths=50_000, seed=100)
# G  = build_call_payoffs(ST, strikes, r, T)

n_paths = 5000
steps_per_year = 252 * 26 # to simulate 252 days with 15 min calibration GARCH
T_years, rate = 1.0, 0.002

sigma0 = np.sqrt(np.var(returns, ddof=1)) 
S_paths = np.empty((n_paths, steps_per_year))
for i in range(n_paths):
    S, _, _ = pm.simulate_prices(S0, steps_per_year, sigma0=sigma0)
    S_paths[i] = S
ST = S_paths[:, -1]
disc = np.exp(-rate * T_years)
G = disc * np.maximum(ST[:, None] - strikes[None, :], 0.0)

wmc = WeightedMC(paths=None, payoffs=G, market_prices=C_mkt) # or C_mkt_bs
wmc.calibrate(lambda0=np.zeros(len(strikes)), max_iter=150)

price_asian = wmc.price(AsianOption(98, 1.0, OptionType.Call).payoff(S_paths))
print(price_asian)











import numpy as np

# p = [ (0.08036276, 0.1, 0.16276168110077988**2, 0.81128139, 1.45265079),
#      (0.0415094, 0.1, 0.16276168110077988**2, 0.99898482, 0.51790126) ]
#p = [(0.28578, 0.05, 0.16276168110077988**2, 0.80959, 8.07617)]
#p = [(0.04108, 0.05, 0.16276168110077988**2, 1.00547, 0.51334)]
p = [(0.07296, 0.5, 0.16276168110077988**2, 0.88159, 1.1806)]
param_sets = p
bs = BlackScholesModel(r=r)
data = data_dict[1.0]
strikes = data["strikes"]
market_vols = data["volatilities"]

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(strikes,market_vols,marker="o",linestyle="--",linewidth=2.5,color="black",label="Market")
plt.plot([99,100,101], market_vols[4:7],linestyle="--", marker='o', linewidth=2.5,color='red',label="Target market point")

model_vols = []
prices = []
for i, (v0, kappa, theta, rho, sigma) in enumerate(param_sets):
    heston = HestonModel(spot=S0,variance=v0,kappa=kappa,theta=theta,sigma=sigma,rho=rho,rate=r,dividend=0.0)
    for K in strikes:
        option = VanillaOption(K, 1.0, OptionType.Call)
        price = heston.Price(option)
        iv = bs.solve_volatility(option=option,S=S0,target_price=price,vol_init=0.2)
        prices.append(price)
        model_vols.append(iv)
    plt.plot(strikes,model_vols,linewidth=1.2, marker='x', label=f"Model", color="orange")

plt.xlabel("Strike")
plt.ylabel("Implied volatility")
plt.title(f"Heston smiles vs market")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Strike")
plt.ylabel("Prices")
plt.title(f"Heston prices vs market")
plt.plot(strikes, data["call"] ,marker="o",linestyle="--",linewidth=2.5,color="black",label="Market")
plt.plot([99,100,101], data["call"][4:7],linestyle="--", marker='o', linewidth=2.5,color='red',label="Target market point")
plt.plot(strikes, prices ,marker="x",linewidth=1.2,color="orange",label="Model")
plt.grid(True)
plt.legend()
plt.show()









from SVI import *
from SSVI import *
import numpy as np
import matplotlib.pyplot as plt


# Usefull functions 
def svi_call_prices(strikes, S0, T, F, a, b, rho, m, sigma):
    prices, vols = [], []
    for K in strikes:
        vol = svi_implied_vol(np.log(K / F), T, a, b, rho, m, sigma)
        vols.append(vol)
        prices.append(bs.PriceCF(VanillaOption(K, T, OptionType.Call), S0, vol))
    return np.array(prices), np.array(vols)


def ssvi_call_prices(strikes, S0, T, F, theta, eta, gamma, rho):
    prices, vols = [], []
    for K in strikes:
        vol = ssvi_implied_vol(np.log(K / F), T, theta, eta, gamma, rho)
        vols.append(vol)
        prices.append(bs.PriceCF(VanillaOption(K, T, OptionType.Call), S0, vol))
    return np.array(prices), np.array(vols)


def risk_neutral_density(strikes, call_prices, r, T):
    strikes = np.array(strikes)
    C = np.array(call_prices)
    dK = strikes[1] - strikes[0]
    d2C = (C[:-2] - 2 * C[1:-1] + C[2:]) / dK**2
    density = np.exp(r * T) * d2C
    return strikes[1:-1], density


def bs_density(K, S0, r, T, sigma):
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    var = sigma**2 * T
    return (1 / (K * np.sqrt(2 * np.pi * var))) * np.exp( -(np.log(K) - mu)**2 / (2 * var))


def kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))



T = 1.0
strikes = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
vols = list(data_dict[1.0]["volatilities"])
forward = S0 * np.exp(r)

# Vegas Black–Scholes (pondérations SVI)
bs = BlackScholesModel(r)
vegas = [bs.Vega(VanillaOption(K, T, OptionType.Call), S0, vol) for K,vol in zip(strikes,vols)]

# Calibration SVI (ou SSVI)
cal = SVIRawCalibrator(strikes, vols, vegas, forward, T)
# cal = SSVICalibrator(strikes, vols, forward, T)
params = cal.calibrate()
print(params)

a, b, rho, m, sigma = params[0]
strikes_dense = strikes
vols_model = [svi_implied_vol(np.log(K / forward), T, a, b, rho, m, sigma)for K in strikes_dense]

# SSVI
# theta, eta, gamma, rho = params[0]
# vols_model = [ssvi_implied_vol(np.log(K / forward), T, theta, eta, gamma, rho) for K in strikes_dense]

# Calculs numériques
n = 1000
K_grid = np.linspace(95, 105, n)

# Prices and SVI density (same for SSVI)
call_prices_svi, svi_vols_dense = svi_call_prices(K_grid, S0, T, forward, a, b, rho, m, sigma)
#call_prices_svi, svi_vols_dense = ssvi_call_prices(K_grid, S0, T, forward, theta, eta, gamma, rho)
K_density, density_svi = risk_neutral_density(K_grid, call_prices_svi, r, T)

# Black–Scholes density
sigma_bs = 0.01 * VIX(1.0)
density_bs = bs_density(K_grid[1:-1], S0, r, T, sigma_bs)

# Heston density
v0, kappa, theta_h, rho_h, sigma_h = [0.0453, 2.18, 0.0265, 0.34, 0.32] # result from Question 3
heston = HestonModel(S0, v0, r, 0.0, kappa, theta_h, sigma_h, rho_h)
prices_heston = [heston.Price(VanillaOption(K, T, OptionType.Call)) for K in K_grid]
_, density_heston = risk_neutral_density(K_grid, prices_heston, r, T)

# KL divergence
kl_svi_bs = kl_divergence(density_svi, density_bs)
kl_heston_bs = kl_divergence(density_heston, density_bs)




# PLOTS (TOUT À LA FIN)
# Smile + fonction g
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(strikes, vols, "o--", lw=2.5, color="black", label="Market")
plt.plot(strikes_dense, vols_model, "o", label="SVI")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
gs = [g(np.log(K / forward), T, a, b, rho, m, sigma) for K in K_grid]
plt.plot(K_grid, gs)
plt.grid(True)
plt.title("Butterfly condition g(k)")
plt.tight_layout()
plt.show()


# Prix Heston
plt.figure(figsize=(6, 4))
plt.plot(K_grid, prices_heston)
plt.title("Heston call prices")
plt.grid(True)
plt.show()


# Densités risque-neutre
plt.figure(figsize=(8, 5))
plt.plot(K_grid[1:-1], density_svi / np.sum(density_svi), label="SVI Density", lw=2)
plt.plot(K_grid[1:-1], density_bs / np.sum(density_bs), "--", label="BS Density", lw=2)
plt.plot(
    K_grid[1:-1],
    density_heston / np.sum(density_heston),
    label="Heston Density",
    lw=2,
)
plt.xlabel("Strike K")
plt.ylabel("Risk-neutral density")
plt.title("SVI vs Black–Scholes vs Heston densities")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Diagnostics
print(f"KL divergence SVI / BS     : {kl_svi_bs}")
print(f"KL divergence Heston / BS  : {kl_heston_bs}")












#params = [0.03565716, 14.99850043, 0.03205743, 0.01872186, 0.57701143]

params = [0.03204502, 0.14590733, 0.03701626, 0.54973209, 0.10291582]
#params = [0.03230518, 5.50339072, 0.03203329, 0.05370702, 0.13130597]
#params = [0.03213713, 1.06694854, 0.03174392, 0.98446514, 0.0784287]

(v0, kappa, theta, rho, sigma) = p[0]
heston_vols = {}

for maturity, data in d.items():
    strikes = data["strikes"]
    market_vols = data["volatilities"]
    forward = data["Forward"]
    model_vols = []
    prices = []
    heston = HestonModel(spot=S0,variance=v0,kappa=kappa,theta=theta,sigma=sigma,rho=rho,rate=r,dividend=0.0)
    bs = BlackScholesModel(r=r)

    for K in strikes:
        option = VanillaOption(K, maturity, OptionType.Call)
        price = heston.Price(option)
        implied_vol = bs.solve_volatility(option=option, S=S0, target_price=price, vol_init=0.2)
        model_vols.append(implied_vol)
        prices.append(price)

    heston_vols[maturity] = model_vols

plt.figure(figsize=(10,6))
for maturity in d.keys():
    strikes = data_dict[maturity]["strikes"]
    plt.plot(strikes,data_dict[maturity]["volatilities"],marker='o',linestyle='--',label=f"Market T={maturity}")
    plt.plot(strikes,heston_vols[maturity],marker='x',linestyle='-',label=f"Heston T={maturity}")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatilities : Market vs Heston")
plt.legend()
plt.grid(True)
plt.show()



