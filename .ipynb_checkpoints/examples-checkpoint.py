import numpy as np
from bisection import bisection
from newton1d import newton1d
from secant1d import secant1d
from newton_system import newton_system
from armijo import armijo
from newton_armijo import newton_armijo

print("=== 1D Root-Finding Examples ===")

# Bisection method
root_bis, it_bis = bisection(lambda x: x**3, -2, 1)
print(f"Bisection method: root = {root_bis:.8f} in {it_bis} iterations")

# Newton's method (1D)
f = lambda x: x * np.exp(-x**2 / 100)
df = lambda x: np.exp(-x**2 / 100) * (1 - 2 * x**2 / 100)
root_newt, it_newt = newton1d(f, df, 2)
print(f"Newton's method (1D): root = {root_newt:.8f} in {it_newt} iterations")

# Secant method (1D)
root_sec, it_sec = secant1d(f, 2, 4)
print(f"Secant method: root = {root_sec:.8f} in {it_sec} iterations")

print("\n=== Multi-Dimensional Root-Finding Example ===")

# Newton's method for systems
F = lambda v: np.array([v[0]*v[1] - 1, v[0]**3 - v[1]**2])
J = lambda v: np.array([[v[1], v[0]], [3*v[0]**2, -2*v[1]]])
root_nd, it_nd = newton_system(F, J, [2, 3])
print(f"Newton's method (system): root = {root_nd}, in {it_nd} iterations")

print("\n=== Armijo Line Search Example ===")

# Armijo line search for quadratic
f_quadratic = lambda x: x[0]**2 + x[1]**2
xk = np.array([1.0, 1.0])
d = np.array([-1.0, -1.0])
grad_fk = np.array([2*xk[0], 2*xk[1]])
t = armijo(f_quadratic, xk, d, grad_fk)
print(f"Armijo step size for quadratic descent: t = {t:.3f}")

print("\n=== Minimization Example (Newton + Armijo) ===")

# Newton + Armijo minimization
f2 = lambda x: (x[0]**2 + x[1]**2) + 0.2 * np.cos(x[0]**2 + x[1])
grad_f2 = lambda x: np.array([
    2*x[0] - 0.4*np.sin(x[0]**2 + x[1])*x[0],
    2*x[1] - 0.2*np.sin(x[0]**2 + x[1])
])
x_min, it_min = newton_armijo(f2, grad_f2, np.array([10.0, 10.0]))
print(f"Newton+Armijo minimization: x* = {x_min}, in {it_min} iterations")
