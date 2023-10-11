import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Betts pg. 543

x0 = np.array([0.0, 0.0, 3.0]) # height, velocity, mass
dt = 0.05

# constants
Tm = 193.044
g = 32.174
sigma = 5.49153484923381010e-5
c = 1580.9425279876559
h0 = 23800

def rocket_dynamics(x, T):

    h = x[0]
    hDot = x[1]
    vDot = (1 / x[2])* (T - sigma*hDot*hDot*np.exp(-h/h0)) - g
    mDot = -T / c
    return hDot, vDot, mDot

def rocket_dynamics_lin(x, T)

    hDot, vDot, mDot = rocket_dynamics(x, T)

    return x + np.a


def solve_max_altitude():

    x = cp.Variable((2, N))  # [pos(1), vel(1)]
    u = cp.Variable((1, N))  # force

    constraints = create_constraints(x, u)
    objective = cp.Minimize(cp.sum_squares(u))

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem.status, x, u


def create_constraints(x, u):

    constraints = []


    return constraints


status, xStar, uStar = solve_min_control_effort()

print(status)

times = np.arange(0.0, tf, dt)

rStar, vStar = xStar.value.T[:, :2].T

plt.plot(times, rStar, 'ro')
plt.axis('equal')
plt.show()

print(times)
print(len(times))
print(dt)