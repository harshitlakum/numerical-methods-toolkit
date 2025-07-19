function [root, num_iter] = bisection(f, a, b, tol, max_iter)
%BISECTION 1D bisection method for root-finding.
%
%   [root, num_iter] = bisection(f, a, b, tol, max_iter)
%
%   Inputs:
%     f        - Function handle, f(x)
%     a, b     - Interval endpoints (f(a)*f(b) < 0)
%     tol      - Tolerance (default: 1e-8)
%     max_iter - Maximum iterations (default: 100)
%
%   Outputs:
%     root     - Estimated root
%     num_iter - Iterations used
%
%   Example:
%     [r, n] = bisection(@(x) x^3, -2, 1)
%
if nargin < 4 || isempty(tol),      tol = 1e-8;    end
if nargin < 5 || isempty(max_iter), max_iter = 100; end

fa = f(a); fb = f(b);
if fa * fb >= 0
    error('f(a) and f(b) must have opposite signs.');
end

for k = 1:max_iter
    xk = (a + b)/2;
    fxk = f(xk);
    if abs(fxk) < tol || (b - a)/2 < tol
        root = xk;
        num_iter = k;
        return
    end
    if fa * fxk > 0
        a = xk; fa = fxk;
    else
        b = xk; fb = fxk;
    end
end
root = xk;
num_iter = max_iter;
end
