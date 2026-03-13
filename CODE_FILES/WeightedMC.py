import numpy as np
from NewtonRaphson import *
from ImpliedVols import *
import matplotlib.pyplot as plt 

class WeightedMC:
    def __init__(self, paths, payoffs, market_prices):
        self.paths = np.asarray(paths)
        self.G = np.asarray(payoffs, dtype=float)     # (n, N)
        self.C = np.asarray(market_prices, dtype=float).reshape(-1)  # (N,)
        self.n, self.N = self.G.shape
        self.lambda_ = None
        self.weights_ = None

    def weights(self, lam):
        lam = np.asarray(lam, dtype=float).reshape(-1)  # (N,)
        x = self.G @ lam                                 # (n,)
        x = x - np.max(x)                                # stability sum exp
        h = np.exp(x)
        return h / np.sum(h)

    def W(self, lam):
        lam = np.asarray(lam, dtype=float).reshape(-1)
        x = self.G @ lam
        x = x - np.max(x)
        return float(np.log(np.sum(np.exp(x))) - np.dot(lam, self.C))

    def grad_W(self, lam):
        q = self.weights(lam)              # (n,)
        return self.G.T @ q - self.C       # (N,)

    def hess_W(self, lam):
        q = self.weights(lam)              # (n,)
        EQ_g = self.G.T @ q                # (N,)
        EQ_gg = (self.G.T * q) @ self.G    # (N,N)
        H = EQ_gg - np.outer(EQ_g, EQ_g)   # Cov
        return H

    def calibrate(self, lambda0=None, tol=1e-8, max_iter=1000):
        if lambda0 is None:
            lambda0 = np.zeros(self.N)
        nr = NewtonRaphson(f=self.W, x0=lambda0,grad=self.grad_W,hessian=self.hess_W,tol=tol,max_iter=max_iter)
        self.lambda_ = nr.solve()
        self.weights_ = self.weights(self.lambda_)
        return self.lambda_

    def price(self, payoff):
        payoff = np.asarray(payoff, dtype=float).reshape(-1)
        return float(np.dot(self.weights_, payoff))
    


    def plot_weight_histogram_by_payoff(
        self,
        payoff,                       # (n,) payoff utilisé pour trier/bin
        bins=60,
        title="Histogram of WMC weights by payoff",
        xlabel="Payoff",
        density=False,                # False: masse par bin ; True: masse / largeur
        show_quantiles=(0.5, 0.9)     # lignes verticales sur quantiles de payoff sous Q~
    ):
        if self.weights_ is None:
            raise RuntimeError("Call calibrate() before plotting weights.")

        payoff = np.asarray(payoff, float).reshape(-1)
        q = np.asarray(self.weights_, float).reshape(-1)
        if payoff.shape[0] != q.shape[0]:
            raise ValueError("payoff and weights_ must have the same length.")

        # bins sur la grille des payoffs
        edges = np.histogram_bin_edges(payoff, bins=bins)
        idx = np.digitize(payoff, edges) - 1
        idx = np.clip(idx, 0, len(edges) - 2)

        mass = np.zeros(len(edges) - 1)
        for k in range(len(mass)):
            mask = (idx == k)
            if np.any(mask):
                mass[k] = q[mask].sum()

        widths = edges[1:] - edges[:-1]
        y = mass / widths if density else mass
        centers = 0.5 * (edges[1:] + edges[:-1])

        plt.figure(figsize=(10, 4))
        plt.bar(centers, y, width=widths, align="center")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Weight density" if density else "Weight mass")
        plt.grid(True, alpha=0.3)

        # quantiles de payoff sous Q~ (avec les poids)
        if show_quantiles is not None and len(show_quantiles) > 0:
            order = np.argsort(payoff)
            p_sorted = payoff[order]
            q_sorted = q[order]
            cdf = np.cumsum(q_sorted)

            for a in show_quantiles:
                a = float(a)
                if not (0.0 < a < 1.0):
                    continue
                kq = np.searchsorted(cdf, a)
                kq = min(max(kq, 0), len(p_sorted) - 1)
                plt.axvline(p_sorted[kq], linestyle="--")
                plt.text(p_sorted[kq], plt.ylim()[1]*0.95, f"{int(100*a)}%", rotation=90, va="top")

        plt.tight_layout()
        plt.show()


    def plot_weight_distribution(self, payoff, title=None):
        """
        Plot the distribution of weights q_i against payoff values,
        sorted by payoff (increasing).
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be calibrated before plotting weights.")

        payoff = np.asarray(payoff, dtype=float).reshape(-1)
        q = self.weights_

        if payoff.shape[0] != q.shape[0]:
            raise ValueError("Payoff and weights must have same length.")

        # sort by payoff
        idx = np.argsort(payoff)
        payoff_sorted = payoff[idx]
        q_sorted = q[idx]

        plt.figure(figsize=(9,4))
        plt.plot(payoff_sorted, q_sorted, lw=1.5)
        plt.xlabel("Payoff value (sorted)")
        plt.ylabel("Weight $q_i$")
        plt.title(title or "Weighted MC – weight distribution vs payoff")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    def plot_cumulative_weights(self, payoff, title=None):
        """
        Plot cumulative sum of weights after sorting by payoff.
        Shows concentration / degeneracy.
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be calibrated before plotting weights.")

        payoff = np.asarray(payoff, dtype=float).reshape(-1)
        q = self.weights_

        idx = np.argsort(payoff)
        payoff_sorted = payoff[idx]
        q_sorted = q[idx]
        q_cum = np.cumsum(q_sorted)

        plt.figure(figsize=(9,4))
        plt.plot(payoff_sorted, q_cum, lw=2)
        plt.axhline(0.5, linestyle="--", color="gray", label="50% mass")
        plt.axhline(0.9, linestyle="--", color="gray", label="90% mass")
        plt.xlabel("Payoff value (sorted)")
        plt.ylabel("Cumulative weight")
        plt.title(title or "Weighted MC – cumulative probability mass")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def calibrated_prices(self):
        """
        Prices of the calibration instruments under the weighted measure Q~
        Returns array of shape (N,)
        """
        if self.weights_ is None:
            raise RuntimeError("Model not calibrated yet.")
        return self.G.T @ self.weights_
    
    def report_calibration(self, names=None, rel=True):
        """
        Print calibration report: Market vs WMC prices
        """
        C_model = self.calibrated_prices()
        C_mkt = self.C

        if names is None:
            names = [f"Instrument {j}" for j in range(self.N)]

        print("\n=== Weighted Monte Carlo calibration ===")
        print(f"{'Instrument':>15} | {'Market':>10} | {'WMC':>10} | {'Error':>10}")
        print("-"*55)

        for j in range(self.N):
            err = C_model[j] - C_mkt[j]
            if rel:
                err = err / max(abs(C_mkt[j]), 1e-12)
                fmt = "{:+.2e}"
            else:
                fmt = "{:+.6f}"

            print(f"{names[j]:>15} | "
                  f"{C_mkt[j]:10.6f} | "
                  f"{C_model[j]:10.6f} | "
                  f"{fmt.format(err):>10}")

        print("-"*55)
        print("Max abs error :", np.max(np.abs(C_model - C_mkt)))
        print("Mean abs error :", np.mean(np.abs(C_model - C_mkt)))

    
    def plot_calibration(self, x=None, xlabel="Instrument", title="WMC calibration"):
        """
        Plot market prices vs calibrated WMC prices
        """
        C_model = self.calibrated_prices()
        C_mkt = self.C

        bs = BlackScholesModel(0.002)
        vol_model = []
        vol_mkt = []
        for i, K in enumerate(range(95,105,1)):
            opt = VanillaOption(K, 1.0, OptionType.Call)
            vol_model.append(bs.solve_volatility(opt, 100, C_model[i]))
            vol_mkt.append(bs.solve_volatility(opt, 100, C_mkt[i]))

        if x is None:
            x = np.arange(self.N)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(7,4))
        plt.plot(x, vol_mkt, "o-", label="Market")
        plt.plot(x, vol_model, "s--", label="WMC")
        plt.xlabel(xlabel)
        plt.ylabel("Price")
        plt.title(title)
        plt.grid(True)
        #plt.ylim((0.0, 0.6))
        plt.legend()
        plt.tight_layout()
        plt.show()