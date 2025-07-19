def bisection(f, a, b, tol=1e-8, max_iter=100):
    """
    Bisection method for finding a root of f(x) = 0 in [a, b].

    Parameters
    ----------
    f : callable
        Function for which the root is sought.
    a : float
        Left endpoint of interval.
    b : float
        Right endpoint of interval.
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
    ValueError
        If f(a) and f(b) do not have opposite signs.

    Example
    -------
    >>> root, n = bisection(lambda x: x**3, -2, 1)
    >>> print(root)
    0.0
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    for k in range(1, max_iter + 1):
        xk = (a + b) / 2.0
        fxk = f(xk)
        if abs(fxk) < tol or (b - a) / 2 < tol:
            return xk, k
        if fa * fxk > 0:
            a, fa = xk, fxk
        else:
            b, fb = xk, fxk
    return xk, max_iter
