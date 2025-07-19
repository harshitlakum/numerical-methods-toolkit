import numpy as np

def newton_system(F, J, x0, tol=1e-8, max_iter=20):
    """
    Newton's method for systems of nonlinear equations.

    Parameters
    ----------
    F : callable
        Function returning array-like F(x) (vector function).
    J : callable
        Function returning array-like Jacobian matrix J(x).
    x0 : array-like
        Initial guess (as a 1D array).
    tol : float, optional
        Tolerance for stopping (default: 1e-8).
    max_iter : int, optional
        Maximum number of iterations (default: 20).

    Returns
    -------
    root : ndarray
        Estimated root as 1D NumPy array.
    num_iter : int
        Number of iterations used.

    Example
    -------
    >>> F = lambda v: np.array([v[0]*v[1] - 1, v[0]**3 - v[1]**2])
    >>> J = lambda v: np.array([[v[1], v[0]], [3*v[0]**2, -2*v[1]]])
    >>> root, n = newton_system(F, J, [2, 3])
    """
    x = np.array(x0, dtype=float)
    for k in range(1, max_iter + 1):
        fval = np.array(F(x))
        if np.linalg.norm(fval, 2) < tol:
            return x, k
        Jval = np.array(J(x))
        dx = np.linalg.solve(Jval, -fval)
        x = x + dx
    return x, max_iter
