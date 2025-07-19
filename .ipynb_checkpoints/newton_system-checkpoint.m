function [root, num_iter] = newton_system(F, J, x0, tol, max_iter)
%NEWTON_SYSTEM Newton's method for systems of nonlinear equations.
%
%   [root, num_iter] = newton_system(F, J, x0, tol, max_iter)
%
%   Inputs:
%     F        - Function handle, F(x) returns column vector
%     J        - Function handle, J(x) returns Jacobian matrix
%     x0       - Initial guess (column vector)
%     tol      - Tolerance (default: 1e-8)
%     max_iter - Maximum iterations (default: 20)
%
%   Outputs:
%     root     - Estimated root (column vector)
%     num_iter - Iterations used
%
%   Example:
%     F = @(v) [v(1)*v(2) - 1; v(1)^3 - v(2)^2];
%     J = @(v) [v(2), v(1); 3*v(1)^2, -2*v(2)];
%     [r, n] = newton_system(F, J, [2; 3])
%
if nargin < 4 || isempty(tol),      tol = 1e-8;    end
if nargin < 5 || isempty(max_iter), max_iter = 20; end

x = x0(:);
for k = 1:max_iter
    fval = F(x);
    if norm(fval, 2) < tol
        root = x;
        num_iter = k;
        return
    end
    Jval = J(x);
    dx = -Jval \ fval;
    x = x + dx;
end
root = x;
num_iter = max_iter;
end
