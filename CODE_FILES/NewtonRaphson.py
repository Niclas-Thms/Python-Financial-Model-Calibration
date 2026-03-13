from Differentiation import *

class NewtonRaphson:
    def __init__(self, f, x0, feasible=None, jacobian=None, grad=None, hessian=None,tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.feasible = feasible
        x0 = np.array([x0], dtype=float)
        self.x = x0.copy()

        # f: R -> R
        if x0.size == 1:
            self.func = lambda xx: np.array([f(xx[0])])
            self.jacobian = (lambda xx: np.array([[jacobian(xx[0])]])) if jacobian else Differentiation.jacobian(self.func)
            return

        # f: R^n -> R 
        f_val = f(self.x)
        if np.isscalar(f_val) or np.ndim(f_val) == 0:
            self.func = grad if grad else Differentiation.gradient(f)
            self.jacobian = hessian if hessian else Differentiation.hessian(f)
            return

        # f: R^n -> R^n
        self.func = f
        self.jacobian = jacobian if jacobian else Differentiation.jacobian(f)

    
    def solve(self, target=None):
        if target is None:
            target = np.zeros_like(self.func(self.x))

        x = self.x.copy() if self.x.size == 1 else self.x.copy()[0]
        prev_norm = np.inf

        for _ in range(self.max_iter):
            fval = np.asarray(self.func(x), float)
            curr_norm = np.linalg.norm(fval)

            if np.linalg.norm(fval - target, ord=2) < self.tol:
                return x
            
            J = np.asarray(self.jacobian(x), float)
            step = np.linalg.solve(J, fval)

            ########################## Question 4 #################################### 
            alpha = 1.0 # linear research to ensure f(x_new) < f(x_prev)
            while alpha > 1e-4:
                x_new = x - alpha * step
                if self.feasible is None or self.feasible(x_new):
                    if np.linalg.norm(self.func(x_new)) < curr_norm:
                        break
                alpha *= 0.5

            # if abs(prev_norm - curr_norm) / max(prev_norm, 1.0) < 1e-6: 
            #     return x_new
            ##########################################################################
            prev_norm = curr_norm
            x = x_new
        print(f"Max iterations reach {self.max_iter}")
        return x