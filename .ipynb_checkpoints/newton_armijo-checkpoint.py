import numpy as np

def newton_armijo(f, grad_f, x0, c1=1e-4, eps_fd=1e-7, tol=1e-8, max_iter=100):
    """
    Newton's method for unconstrained minimization with Armijo line search.

    Parameters
    ----------
    f : callable
        Objective function, f(x).
    grad_f : callable
        Gradient function, grad_f(x) returns 1D array.
    x0 : array-like
        Initial guess (1D array).
    c1 : float, optional
        Armijo parameter (default: 1e-4).
    eps_fd : float, optional
        Step size for finite-difference Hessian (default: 1e-7).
    tol : float, optional
        Stopping tolerance for gradient norm (default: 1e-8).
    max_iter : int, optional
        Maximum iterations (default: 100).

    Returns
    -------
    x : ndarray
        Estimated minimizer (1D array).
    num_iter : int
        Number of iterations used.

    Example
    -------
    >>> f = lambda x: x[0]**2 + x[1]**2
    >>> grad_f = lambda x: np.array([2*x[0], 2*x[1]])
    >>> x_min, n = newton_armijo(f, grad_f, np.array([1.0, 1.0]))
    """
    def hessian_fd(grad_f, x, eps=1e-7):
        n = len(x)
        H = np.zeros((n, n))
        g0 = grad_f(x)
        for j in range(n):
            x_eps = np.copy(x)
            x_eps[j] += eps
            g1 = grad_f(x_eps)
            H[:, j] = (g1 - g0) / eps
        return H

    x = np.copy(x0)
    for k in range(1, max_iter + 1):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x, k
        H = hessian_fd(grad_f, x, eps_fd)
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Hessian not invertible at iteration", k)
            return x, k
        t = armijo(f, x, d, g, c1)
        x_new = x + t * d
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter
