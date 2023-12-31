import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

x0 = np.array([0.0, 0.0])
x1 = np.array([1.0, 0.0])

N = 40
tf = 10.0
dt = tf / N

# dynamics
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])


def solve_min_control_effort():

    x = cp.Variable((2, N))  # [pos(1), vel(1)]
    u = cp.Variable((1, N))  # force

    constraints = create_constraints(x, u)
    objective = cp.Minimize(cp.sum_squares(u))

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem.status, x, u


def create_constraints(x, u):

    constraints = []
    constraints.append(x[:, 0] == x0)  # initial state
    constraints.append(x[:, N-1] == x1)  # final state

    # dynamics constraints
    for t in range(N - 1):
        constraints.append(x[:, t + 1] == x[:, t] + (A @ x[:, t] + B @ u[:, t]) * dt)

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