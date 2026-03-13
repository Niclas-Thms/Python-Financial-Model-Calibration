import numpy as np
from nelder_mead import nelder_mead

def first_derivative_ssvi(k, theta, eta, gamma, rho):
    phi_theta = phi(theta, eta, gamma)
    A = phi_theta * k + rho
    D = np.sqrt(A*A + 1.0 - rho*rho)
    return 0.5 * theta * phi_theta * (rho + A / D)

def second_derivative_ssvi(k, theta, eta, gamma, rho):
    phi_theta = phi(theta, eta, gamma)
    A = phi_theta * k + rho
    D = np.sqrt(A*A + 1.0 - rho*rho)
    return 0.5 * theta * (phi_theta**2) * (1.0 - rho*rho) / (D**3)

def g_ssvi(k, T, theta, eta, gamma, rho):
    w   = ssvi_implied_vol(k, T, theta, eta, gamma, rho)**2 * T
    w_  = first_derivative_ssvi(k, theta, eta, gamma, rho)
    w__ = second_derivative_ssvi(k, theta, eta, gamma, rho)
    return (1.0 - 0.5 * k * w_ / w)**2 - 0.25 * (w_**2) * (1.0 / w + 0.25) + 0.5 * w__


def phi(theta, eta, gamma):
    return eta / (theta**gamma * (1 + theta)**(1 - gamma))

def ssvi_implied_vol(k, T, theta, eta, gamma, rho):
    phi_theta = phi(theta, eta, gamma)
    return np.sqrt(0.5 * theta * (1 + rho * phi_theta * k + 
                                  np.sqrt((phi_theta * k + rho)**2 + 1 - rho**2)) / T)

class SSVICalibrator:
    def __init__(self, strikes, vols, forward, T):
        self.k = np.log(np.array(strikes) / forward)
        self.vols = np.array(vols)
        self.T = T
        self.theta_init = (vols[len(vols)//2]**2) * T  # ATM approx

    def objective(self, x):
        theta, eta, gamma, rho = x

        # contraintes théoriques
        if theta <= 0 or abs(rho) >= 1 or gamma < 0 or gamma > 0.5:
            return np.inf
        if eta <= 0 or eta > 2/np.sqrt(1 + np.abs(rho)):
            return np.inf

        vol_model = ssvi_implied_vol(self.k, self.T, theta, eta, gamma, rho)
        return np.mean((vol_model - self.vols)**2)

    def calibrate(self):
        point1 = np.array([0.2, 1, 0.01, -0.5])
        point2 = np.array([0.1, 1.5, 0.5, -0.2])
        point3 = np.array([0.15, 0.5, 0.25, -0.7])
        point4 = np.array([0.25, 0.7, 0.12, -0.1])
        point5 = np.array([0.18, 1.4, 0.37, -0.8])

        x0 = [point1, point2, point3, point4, point5]
        res = nelder_mead(self.objective, simplex=x0)
        return res
    




    