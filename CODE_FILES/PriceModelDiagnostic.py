from TimeSeriesStats import TimeSeriesStats
from PriceModel import PriceModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t



class Diagnostics:

    @staticmethod
    def run(r, model=PriceModel, nlags=30):
        mu, omega, alpha, beta, nu = model.params_
        u, _ = PriceModelDiagnostics.standardized_residuals(r, model)
        print("\n[Calibrated parameters]")
        print(f"mu={mu:.4g}, omega={omega:.4g}, "
              f"alpha={alpha:.4g}, beta={beta:.4g}, nu={nu:.4g}")
        print(f"Persistence alpha+beta={alpha+beta:.4f}")
        print(f"log-likelihood={model.loglik_:.3f}")
        TimeSeriesStats.plot_acf(u, nlags, "ACF of standardized residuals u_t")
        TimeSeriesStats.plot_acf(u**2, nlags, "ACF of squared residuals u_t^2")
        coverage = PriceModelDiagnostics.coverage(r, model, p=0.90)
        print(f"\n[Coverage]")
        print(f"P(r_t ∈ [q5%, q95%]) = {coverage:.3f} (target ≈ 0.90)")
        PriceModelDiagnostics.qqplot(u, nu) 


class PriceModelDiagnostics:
    @staticmethod
    def standardized_residuals(r, model: PriceModel):
        mu, omega, alpha, beta, nu = model.params_
        n = len(r)
        eps = r - mu
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r, ddof=1)
        for t_ in range(1, n):
            sigma2[t_] = omega + alpha * eps[t_-1]**2 + beta * sigma2[t_-1]
        u = eps / np.sqrt(sigma2)
        return u, sigma2

    @staticmethod
    def coverage(r, model: PriceModel, p=0.90):
        _, sigma2 = PriceModelDiagnostics.standardized_residuals(r, model)
        mu, _, _, _, nu = model.params_
        ql = mu + np.sqrt(sigma2) * t.ppf((1-p)/2, nu)
        qh = mu + np.sqrt(sigma2) * t.ppf(1-(1-p)/2, nu)
        return np.mean((r >= ql) & (r <= qh))

    @staticmethod
    def qqplot(u, nu):
        u = np.sort(u[np.isfinite(u)])
        n = len(u)
        p = (np.arange(1, n+1) - 0.5) / n
        q = t.ppf(p, nu)
        plt.figure(figsize=(5,5))
        plt.scatter(q, u, s=10)
        lim = max(abs(q).max(), abs(u).max())
        plt.plot([-lim, lim], [-lim, lim], '--')
        plt.xlabel("Theoretical quantiles (Student-t)")
        plt.ylabel("Empirical quantiles")
        plt.title("QQ-plot standardized residuals")
        plt.grid(True)
        plt.show()


