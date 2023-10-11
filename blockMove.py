import numpy as np
import cvxpy as cp


x0 = np.array([0.0, 0.0])
x1 = np.array([1.0, 0.0])

e1 = np.array([1, 0]).T
e2 = np.array([0, 1]).T

N = 50
tf = 10.0
dt = tf / N

# dynamics
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0, 1.0]])


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

        statenow = x[:, t]
        constraints.append(x[:, t + 1] == x[:, t] + (A @ x[:, t] + B.transpose() @ u[:, t]) * dt)

    return constraints


status, _, _ = solve_min_control_effort()

print(status)
