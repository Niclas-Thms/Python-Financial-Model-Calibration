import numpy as np
from math import pi, gamma


def simplex_diameter(simplex):
    max_dist = 0.0
    for i in range(len(simplex)):
        for j in range(i + 1, len(simplex)):
            dist = np.linalg.norm(simplex[i] - simplex[j])
            max_dist = max(max_dist, dist)
    return max_dist

def hypersphere_volume(d, radius):
    return (pi ** (d / 2)) / gamma(d / 2 + 1) * radius ** d

def nelder_mead(objective, simplex, max_iter=200, adaptation=True,
                 tol_volume=1e-8, alpha=1.0, beta=2.0, gamma=0.5, delta=0.5):
    simplex = [np.array(x, dtype=float) for x in simplex]
    d = len(simplex[0])  
    if adaptation:
        beta, gamma, delta = 1 + 2/d, 0.75 - 0.5 * 1/d, 1 - 1/d  #article adaptation
    m = d + 1      

    values = [objective(x) for x in simplex]
    for _ in range(max_iter):

        idx = np.argsort(values)
        simplex = [simplex[i] for i in idx]
        values  = [values[i]  for i in idx]

        D = simplex_diameter(simplex)
        volume = hypersphere_volume(d, 0.5 * D)
        if volume < tol_volume:
            print(_)
            break

        best, worst  = simplex[0], simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = objective(xr)
        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
            continue

        # Expansion
        if fr < values[0]:
            xe = centroid + beta * (xr - centroid)
            fe = objective(xe)
            simplex[-1] = xe if fe < fr else xr
            values[-1]  = min(fe, fr)
            continue

        # Contraction
        xc = centroid + gamma * (worst - centroid)
        fc = objective(xc)
        if fc < values[-1]:
            simplex[-1] = xc
            values[-1] = fc
            continue

        # Shrink
        for i in range(1, m):
            simplex[i] = best + delta * (simplex[i] - best)
            values[i] = objective(simplex[i])

    return simplex[0], values[0]