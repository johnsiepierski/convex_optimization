import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# parameters
x0 = np.array([1.1, 0.0, 0.0, 1/np.sqrt(1.1)]) # r, theta, vr, vt
N = 40
tf = 10.0
dt = tf / N

# variables
x = cp.Variable((4, N))
beta = cp.Variable((1, N))  # direction of thrust


# unit vectors
e0 = np.array([1, 0, 0, 0]).T #r
e1 = np.array([0, 1, 0, 0]).T #theta
e2 = np.array([0, 0, 1, 0]).T #vr
e3 = np.array([0, 0, 0, 1]).T #vt


def compute_dstate(x, u):

    r = e0 @ x
    vr = e2 @ x
    vt = e3 @ x

    rDot = vr
    thetaDot = vt / r

    vrDot = vt**2 / r - 1 / r**2 + 0.01 * np.sin(u)
    vtDot = -(vt*vr)/r + 0.01*np.cos(u)

    A = np.zeros([rDot, thetaDot, vrDot, vtDot])

    return A


def objective(x_f):
    r = e0 @ x_f
    vr = e2 @ x_f
    vt = e3 @ x_f

    return 1/r - 0.5*(vr**2+vt**2)


# constraints
constraints = []

for t in range(N - 1):

    constraints.append(e0 @ x[:, t] >= 0.5)
    constraints.append(e0 @ x[:, t] <= 5)

    constraints.append(e1 @ x[:, t] >= 0.0)
    constraints.append(e1 @ x[:, t] <= 8*np.pi)

    constraints.append(e2 @ x[:, t] >= -10.0)
    constraints.append(e2 @ x[:, t] <= 10)

    constraints.append(e3 @ x[:, t] >= 0.0)
    constraints.append(e3 @ x[:, t] <= 10.0)

    # thrust direction constraints
    constraints.append(beta[:, t] >= -np.pi / 2.0)
    constraints.append(beta[:, t] <= np.pi / 2.0)

    # dynamics constraints
    constraints.append(x[:, t + 1] == x[:, t] + compute_dstate(x[:, t], beta[:, t]) * dt)

# objective
objective = cp.Minimize(objective(x[:, N-1]))

problem = cp.Problem(objective, constraints)
problem.solve()

print(problem.status())

