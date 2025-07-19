function [x, num_iter] = newton_armijo(f, grad_f, x0, c1, eps_fd, tol, max_iter)
%NEWTON_ARMIJO Newton's method for unconstrained minimization with Armijo line search.
%
%   [x, num_iter] = newton_armijo(f, grad_f, x0, c1, eps_fd, tol, max_iter)
%
%   Inputs:
%     f        - Objective function handle, f(x)
%     grad_f   - Gradient function handle, grad_f(x)
%     x0       - Initial guess (column vector)
%     c1       - Armijo parameter (default: 1e-4)
%     eps_fd   - Finite difference step for Hessian (default: 1e-7)
%     tol      - Gradient norm tolerance (default: 1e-8)
%     max_iter - Maximum iterations (default: 100)
%
%   Outputs:
%     x        - Estimated minimizer (column vector)
%     num_iter - Iterations used
%
%   Example:
%     f = @(x) x(1)^2 + x(2)^2;
%     grad_f = @(x) [2*x(1); 2*x(2)];
%     [x_min, n] = newton_armijo(f, grad_f, [1;1]);
%
if nargin < 4 || isempty(c1),      c1 = 1e-4;    end
if nargin < 5 || isempty(eps_fd),  eps_fd = 1e-7;end
if nargin < 6 || isempty(tol),     tol = 1e-8;   end
if nargin < 7 || isempty(max_iter),max_iter = 100;end

x = x0(:);
for k = 1:max_iter
    g = grad_f(x);
    if norm(g) < tol
        num_iter = k;
        return
    end
    % Finite-difference Hessian
    n = length(x);
    H = zeros(n,n);
    g0 = grad_f(x);
    for j = 1:n
        x_eps = x; x_eps(j) = x_eps(j) + eps_fd;
        g1 = grad_f(x_eps);
        H(:,j) = (g1 - g0) / eps_fd;
    end
    d = -H\g;
    t = armijo(f, x, d, g, c1);
    x_new = x + t*d;
    if norm(x_new - x) < tol
        x = x_new;
        num_iter = k;
        return
    end
    x = x_new;
end
num_iter = max_iter;
end
