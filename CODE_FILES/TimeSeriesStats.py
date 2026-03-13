import numpy as np
import math
import matplotlib.pyplot as plt

class TimeSeriesStats:
    @staticmethod
    def acf(x, nlags=30):
        x = np.array(x, dtype=float)
        x = x - x.mean()
        den = np.dot(x, x)
        out = np.empty(nlags + 1)
        out[0] = 1.0
        for k in range(1, nlags + 1):
            out[k] = np.dot(x[k:], x[:-k]) / den
        return out

    @staticmethod
    def jarque_bera(x):
        x = np.array(x, dtype=float)
        n = len(x)
        m = x.mean()
        s2 = np.mean((x - m)**2)
        if s2 <= 0:
            return np.nan, np.nan, np.nan, np.nan
        s = np.mean(((x - m)/math.sqrt(s2))**3)
        k = np.mean(((x - m)/math.sqrt(s2))**4)
        jb = n/6 * (s*s + 0.25*(k - 3)**2)
        return jb, s, k, k - 3

    @staticmethod
    def plot_acf(x, nlags=30, title="ACF"):
        x = np.asarray(x, dtype=float)
        vals = TimeSeriesStats.acf(x, nlags)
        n = len(x)
        conf = 1.96 / math.sqrt(n)  # approx 95%
        lags = np.arange(nlags + 1)
        plt.figure(figsize=(12, 5))
        markerline, stemlines, baseline = plt.stem(lags, vals, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
        plt.setp(stemlines, linewidth=2)
        plt.setp(markerline, markersize=7)
        plt.setp(baseline, linewidth=1)
        plt.axhline(conf, linestyle="--", linewidth=1.5)
        plt.axhline(-conf, linestyle="--", linewidth=1.5)
        plt.axhline(0.0, linewidth=1)
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.xlim(-0.5, nlags + 0.5)
        plt.grid(True, which="both", axis="both", linestyle=":", linewidth=1)
        plt.tight_layout()
        plt.show()