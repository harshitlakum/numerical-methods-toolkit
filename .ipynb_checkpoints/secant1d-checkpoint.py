def secant1d(f, x0, x1, tol=1e-8, max_iter=100):
    """
    Secant method for 1D root-finding.

    Parameters
    ----------
    f : callable
        Function for which the root is sought.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
    tol : float, optional
        Tolerance for the root (default: 1e-8).
    max_iter : int, optional
        Maximum number of iterations (default: 100).

    Returns
    -------
    root : float
        Estimated root.
    num_iter : int
        Number of iterations used.

    Raises
    ------
    RuntimeError
        If denominator becomes zero.

    Example
    -------
    >>> f = lambda x: x * np.exp(-x**2/100)
    >>> root, n = secant1d(f, 2, 4)
    """
    for k in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)
        if abs(f1) < tol:
            return x1, k
        denom = f1 - f0
        if denom == 0:
            raise RuntimeError(f"Zero denominator encountered at iteration {k}.")
        x_new = x1 - f1 * (x1 - x0) / denom
        x0, x1 = x1, x_new
    return x1, max_iter
