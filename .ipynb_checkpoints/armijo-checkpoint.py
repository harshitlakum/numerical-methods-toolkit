def armijo(f, xk, d, grad_fk, c1=1e-4, t_init=1.0, beta=0.5, max_iter=10):
    """
    Armijo (backtracking) line search for step length selection in descent methods.

    Parameters
    ----------
    f : callable
        Objective function, f(x).
    xk : array-like
        Current point (1D array or float).
    d : array-like
        Descent direction (same shape as xk).
    grad_fk : array-like
        Gradient at xk (same shape as xk).
    c1 : float, optional
        Armijo parameter, 0 < c1 < 1 (default: 1e-4).
    t_init : float, optional
        Initial step size (default: 1.0).
    beta : float, optional
        Reduction factor (0 < beta < 1, default: 0.5).
    max_iter : int, optional
        Maximum backtracking steps (default: 10).

    Returns
    -------
    t : float
        Step length satisfying Armijo condition.

    Example
    -------
    >>> f = lambda x: (x[0]**2 + x[1]**2)
    >>> grad_fk = np.array([2*xk[0], 2*xk[1]])
    >>> t = armijo(f, np.array([1,1]), np.array([-1,-1]), grad_fk)
    """
    t = t_init
    fxk = f(xk)
    for _ in range(max_iter):
        new_x = xk + t * d
        if f(new_x) <= fxk + c1 * t * float(np.dot(grad_fk, d)):
            return t
        t *= beta
    return t  # Returns last t, even if Armijo condition not met
