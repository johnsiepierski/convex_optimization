import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


x0 = np.array([0.0, 0.0])

e1 = np.array([1, 0, 0, 0]).T
e2 = np.array([0, 1, 0, 0]).T

N = 50

dmax = 100
dmin = -100

umax = 10
umin = -10




def create_constraints(x, u):
    constraints = []
    # boundary constraints
    constraints.append(x[:, 0] == x0)
    constraints.append(x[1, N-1] == np.pi)

    # max cart displacement
    # constraints.append(x[0, N-1] <= dmax)
    # constraints.append(x[0, N-1] >= dmin)

    # max force
    constraints.append(u[N - 1] <= umax)
    constraints.append(u[N - 1] >= umin)

    # dynamics constraints
    for t in range(N - 1):

        constraints.append(x[0, t+1], =)


def cartpole_dynamics_autogen(q, dq, u, m1, m2, g, l):

    t2 = np.cos(q)
    t3 = np.sin(q)
    t4 = t2**2
    t5 = m1+m2-m2*t4
    t6 = 1.0/t5
    t7 = dq**2
    ddx = t6*(u+g*m2*t2*t3+l*m2*t3*t7)
    ddq = -(t6*(t2*u+g*m1*t3+g*m2*t3+l*m2*t2*t3*t7))/l
    return ddx, ddq



