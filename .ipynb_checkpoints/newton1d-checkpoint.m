function [root, num_iter] = newton1d(f, df, x0, tol, max_iter)
%NEWTON1D  Newton's method for 1D root-finding.
%
%   [root, num_iter] = newton1d(f, df, x0, tol, max_iter)
%
%   Inputs:
%     f        - Function handle, f(x)
%     df       - Derivative handle, df(x)
%     x0       - Initial guess
%     tol      - Tolerance (default: 1e-8)
%     max_iter - Maximum iterations (default: 100)
%
%   Outputs:
%     root     - Estimated root
%     num_iter - Iterations used
%
%   Example:
%     f = @(x) x .* exp(-x.^2/100);
%     df = @(x) exp(-x.^2/100) .* (1 - 2*x.^2/100);
%     [r, n] = newton1d(f, df, 2)
%
if nargin < 4 || isempty(tol),      tol = 1e-8;    end
if nargin < 5 || isempty(max_iter), max_iter = 100; end

x = x0;
for k = 1:max_iter
    fx = f(x);
    dfx = df(x);
    if abs(fx) < tol
        root = x;
        num_iter = k;
        return
    end
    if dfx == 0
        error('Zero derivative encountered at iteration %d.', k);
    end
    x = x - fx / dfx;
end
root = x;
num_iter = max_iter;
end
