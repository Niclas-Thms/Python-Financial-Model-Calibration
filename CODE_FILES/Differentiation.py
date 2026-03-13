import numpy as np

class Differentiation:
    @staticmethod
    def jacobian(f, eps=1e-6):
        def J(x):
            x = np.array(x, dtype=float)
            f0 = np.array(f(x), dtype=float)
            m, n = len(f0), len(x)
            J = np.zeros((m, n))
            for j in range(n):
                x1 = x.copy()
                x1[j] += eps
                J[:, j] = (np.array(f(x1)) - f0) / eps
            return J
        return J

    @staticmethod
    def gradient(f, eps=1e-6):
        def g(x):
            x = np.array(x, dtype=float)
            n = len(x)
            grad = np.zeros(n)
            f0 = f(x)
            for j in range(n):
                x1 = x.copy()
                x1[j] += eps
                grad[j] = (f(x1) - f0) / eps
            return grad
        return g

    @staticmethod
    def hessian(f, eps=1e-4):
        def H(x):
            x = np.array(x, dtype=float)
            n = len(x)
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    e_i = np.zeros(n)
                    e_j = np.zeros(n)
                    e_i[i] = eps
                    e_j[j] = eps
                    H[i, j] = (f(x + e_i + e_j) - f(x + e_i) - f(x + e_j) + f(x)) / (eps ** 2)
            return H
        return H