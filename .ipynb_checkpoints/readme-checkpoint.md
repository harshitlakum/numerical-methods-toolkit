# Numerical Methods Toolkit

A unified collection of classic numerical algorithms for root-finding and unconstrained minimization in **Python** and **MATLAB/Octave**. This toolkit is designed for scientific computing, homework, research, or teaching.

---

## Features

- **1D Root-Finding:** Bisection, Newton's, and Secant methods
- **Multi-Dimensional Root-Finding:** Newton's method for systems
- **Unconstrained Minimization:** Newton's method with Armijo line search
- **Armijo Line Search:** Adaptive step length selection for optimization
- Well-documented, modular code in both Python and MATLAB/Octave

---

## Repository Structure

```

numerical-methods-toolkit/
│
├── python/
│   ├── bisection.py
│   ├── newton1d.py
│   ├── secant1d.py
│   ├── newton\_system.py
│   ├── armijo.py
│   ├── newton\_armijo.py
│   └── examples.py
│
├── matlab/
│   ├── bisection.m
│   ├── newton1d.m
│   ├── secant1d.m
│   ├── newton\_system.m
│   ├── armijo.m
│   ├── newton\_armijo.m
│   └── examples.m
│
├── README.md



## Provided Methods

| Method               | Python file       | MATLAB file      | Description                          |
| -------------------- | ----------------- | ---------------- | ------------------------------------ |
| Bisection (1D)       | bisection.py      | bisection.m      | Robust root-finding in 1D            |
| Newton's (1D)        | newton1d.py       | newton1d.m       | Quadratic 1D root-finding            |
| Secant (1D)          | secant1d.py       | secant1d.m       | Derivative-free 1D root-finding      |
| Newton's (nD/system) | newton\_system.py | newton\_system.m | Multi-D root-finding                 |
| Armijo line search   | armijo.py         | armijo.m         | Step length selection (optimization) |
| Newton + Armijo (nD) | newton\_armijo.py | newton\_armijo.m | Unconstrained minimization           |

---

