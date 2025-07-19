def newton1d(f, df, x0, tol=1e-8, max_iter=100):
    """
    Newton's method for 1D root-finding.

    Parameters
    ----------
    f : callable
        Function for which the root is sought.
    df : callable
        Derivative of f.
    x0 : float
        Initial guess.
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
        If the derivative becomes zero.

    Example
    -------
    >>> f = lambda x: x * np.exp(-x**2/100)
    >>> df = lambda x: np.exp(-x**2/100) * (1 - 2*x**2/100)
    >>> root, n = newton1d(f, df, 2)
    """
    x = x0
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x, k
        if dfx == 0:
            raise RuntimeError(f"Zero derivative encountered at iteration {k}.")
        x = x - fx / dfx
    return x, max_iter
