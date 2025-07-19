function t = armijo(f, xk, d, grad_fk, c1, t_init, beta, max_iter)
%ARMIJO Armijo (backtracking) line search for step length selection.
%
%   t = armijo(f, xk, d, grad_fk, c1, t_init, beta, max_iter)
%
%   Inputs:
%     f        - Function handle, f(x)
%     xk       - Current point (vector or scalar)
%     d        - Descent direction (vector or scalar)
%     grad_fk  - Gradient at xk (vector or scalar)
%     c1       - Armijo parameter, 0 < c1 < 1 (default: 1e-4)
%     t_init   - Initial step size (default: 1.0)
%     beta     - Reduction factor, 0 < beta < 1 (default: 0.5)
%     max_iter - Maximum backtracking steps (default: 10)
%
%   Output:
%     t        - Step length satisfying Armijo condition
%
%   Example:
%     f = @(x) x(1)^2 + x(2)^2;
%     grad_fk = [2*xk(1); 2*xk(2)];
%     t = armijo(f, [1;1], [-1;-1], grad_fk)
%
if nargin < 5 || isempty(c1),      c1 = 1e-4;    end
if nargin < 6 || isempty(t_init),  t_init = 1.0; end
if nargin < 7 || isempty(beta),    beta = 0.5;   end
if nargin < 8 || isempty(max_iter),max_iter = 10;end

t = t_init;
fxk = f(xk);
for j = 1:max_iter
    new_x = xk + t*d;
    if f(new_x) <= fxk + c1*t*(grad_fk'*d)
        return
    end
    t = t*beta;
end
% If condition never met, return last t
end
