disp('=== 1D Root-Finding Examples ===');

% Bisection method
[root_bis, it_bis] = bisection(@(x) x^3, -2, 1);
fprintf('Bisection method: root = %.8f in %d iterations\n', root_bis, it_bis);

% Newton's method (1D)
f = @(x) x .* exp(-x.^2/100);
df = @(x) exp(-x.^2/100) .* (1 - 2*x.^2/100);
[root_newt, it_newt] = newton1d(f, df, 2);
fprintf('Newton''s method (1D): root = %.8f in %d iterations\n', root_newt, it_newt);

% Secant method (1D)
[root_sec, it_sec] = secant1d(f, 2, 4);
fprintf('Secant method: root = %.8f in %d iterations\n', root_sec, it_sec);

disp(' ');
disp('=== Multi-Dimensional Root-Finding Example ===');

% Newton's method for systems
F = @(v) [v(1)*v(2) - 1; v(1)^3 - v(2)^2];
J = @(v) [v(2), v(1); 3*v(1)^2, -2*v(2)];
[root_nd, it_nd] = newton_system(F, J, [2; 3]);
fprintf('Newton''s method (system): root = [%g, %g], in %d iterations\n', root_nd(1), root_nd(2), it_nd);

disp(' ');
disp('=== Armijo Line Search Example ===');

% Armijo line search for quadratic
f_quadratic = @(x) x(1)^2 + x(2)^2;
xk = [1; 1];
d = [-1; -1];
grad_fk = [2*xk(1); 2*xk(2)];
t = armijo(f_quadratic, xk, d, grad_fk);
fprintf('Armijo step size for quadratic descent: t = %.3f\n', t);

disp(' ');
disp('=== Minimization Example (Newton + Armijo) ===');

% Newton + Armijo minimization
f2 = @(x) (x(1)^2 + x(2)^2) + 0.2 * cos(x(1)^2 + x(2));
grad_f2 = @(x) [2*x(1) - 0.4*sin(x(1)^2 + x(2))*x(1); 2*x(2) - 0.2*sin(x(1)^2 + x(2))];
[x_min, it_min] = newton_armijo(f2, grad_f2, [10; 10]);
fprintf('Newton+Armijo minimization: x* = [%g, %g], in %d iterations\n', x_min(1), x_min(2), it_min);
