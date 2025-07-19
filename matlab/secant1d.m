function [root, num_iter] = secant1d(f, x0, x1, tol, max_iter)
%SECANT1D  Secant method for 1D root-finding.
%
%   [root, num_iter] = secant1d(f, x0, x1, tol, max_iter)
%
%   Inputs:
%     f        - Function handle, f(x)
%     x0       - First initial guess
%     x1       - Second initial guess
%     tol      - Tolerance (default: 1e-8)
%     max_iter - Maximum iterations (default: 100)
%
%   Outputs:
%     root     - Estimated root
%     num_iter - Iterations used
%
%   Example:
%     f = @(x) x .* exp(-x.^2/100);
%     [r, n] = secant1d(f, 2, 4)
%
if nargin < 4 || isempty(tol),      tol = 1e-8;    end
if nargin < 5 || isempty(max_iter), max_iter = 100; end

for k = 1:max_iter
    f0 = f(x0);
    f1 = f(x1);
    if abs(f1) < tol
        root = x1;
        num_iter = k;
        return
    end
    denom = f1 - f0;
    if denom == 0
        error('Zero denominator encountered at iteration %d.', k);
    end
    x_new = x1 - f1 * (x1 - x0) / denom;
    x0 = x1;
    x1 = x_new;
end
root = x1;
num_iter = max_iter;
end
