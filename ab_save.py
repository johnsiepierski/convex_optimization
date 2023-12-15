import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24    # mass of the Earth in kg
R = 6.371e6     # radius of the Earth in m

# Problem definition
N = 50  # number of control intervals
T = 360.0  # end time in seconds
dt = T / N  # length of one control interval

opti = ca.Opti()

r = opti.variable(3, N+1)  # position
v = opti.variable(3, N+1)  # velocity
u = opti.variable(3, N)    # control input (acceleration due to thrust)

# Initial conditions
r0 = np.array([6771000.0, 0, 0]) # initial position
v0 = np.array([0, 7672.4904132836045, 0])  # initial velocity

rf = np.array([6.20464485e+06, 2.73867844e+06, 0.0])  # final position
vf = np.array([-3100.85365, 7331.52220, 0.0])  # final velocity

opti.subject_to(r[:, 0] == r0)
opti.subject_to(v[:, 0] == v0)
opti.subject_to(r[:, -1] == rf)
opti.subject_to(v[:, -1] == vf)

# Dynamics constraints
for i in range(N):
    r_mid = (r[:, i] + r[:, i+1]) / 2
    dist_mid = ca.sqrt(ca.dot(r_mid, r_mid))
    g_mid = -G * M / dist_mid**3 * r_mid  # gravitational acceleration

    opti.subject_to(r[:, i+1] == r[:, i] + dt*v[:, i] + 0.5 * dt**2 * g_mid)
    opti.subject_to(v[:, i+1] == v[:, i] + dt*(g_mid + u[:, i]))

# Control constraints
max_thrust_acceleration = 30000.0  # m/s^2
min_thrust_acceleration = 0.10  # m/s^2

for i in range(N):
    opti.subject_to(ca.sqrt(ca.mtimes(u[:, i].T, u[:, i])) <= max_thrust_acceleration)
    opti.subject_to(ca.sqrt(ca.mtimes(u[:, i].T, u[:, i])) >= min_thrust_acceleration)

# Objective
objective = ca.sumsqr(u) * dt
opti.minimize(objective)

# Initial guess
opti.set_initial(r, np.linspace(r0, rf, N+1).T)
opti.set_initial(v, np.zeros((N+1, 3)).T)
opti.set_initial(u, np.ones((N, 3)).T * 5.0)     # Small constant thrust

# Solve
opti.solver('ipopt')
sol = opti.solve()

# Extract solution
r_opt = sol.value(r)
v_opt = sol.value(v)
u_opt = sol.value(u)
thrust_magnitude = np.linalg.norm(u_opt, axis=0)


# Calculate total delta-v
delta_v_total = np.sum(np.linalg.norm(u_opt, axis=0)) * dt

# Print total delta-v
print(f"Total delta-v: {delta_v_total:.2f} m/s")

# Plotting
time = np.linspace(0, T, N+1)
time_u = np.linspace(0, T-dt, N)

plt.figure(figsize=(15, 5))

# Position plot
plt.subplot(2, 2, 1)
plt.plot(time, r_opt[0, :], label='x')
plt.plot(time, r_opt[1, :], label='y')
plt.plot(time, r_opt[2, :], label='z')
plt.plot(time[-1], rf[0], 'r*', markersize=10)  # Final position target marker for x
plt.plot(time[-1], rf[1], 'g*', markersize=10)  # Final position target marker for y
plt.plot(time[-1], rf[2], 'b*', markersize=10)  # Final position target marker for z
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

# Velocity plot
plt.subplot(2, 2, 2)
plt.plot(time, v_opt[0, :], label='vx')
plt.plot(time, v_opt[1, :], label='vy')
plt.plot(time, v_opt[2, :], label='vz')
plt.plot(time[-1], vf[0], 'r*', markersize=10)  # Final velocity target marker for vx
plt.plot(time[-1], vf[1], 'g*', markersize=10)  # Final velocity target marker for vy
plt.plot(time[-1], vf[2], 'b*', markersize=10)  # Final velocity target marker for vz
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Thrust plot
plt.subplot(2, 2, 3)
plt.step(time_u, u_opt[0, :], label='ux')
plt.step(time_u, u_opt[1, :], label='uy')
plt.step(time_u, u_opt[2, :], label='uz')
plt.title('Thrust Components vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()

# Thrust magnitude plot
plt.subplot(2, 2, 4)
plt.step(time_u, thrust_magnitude, label='Thrust Magnitude')
plt.title('Thrust Magnitude vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust Magnitude (N)')
plt.legend()

plt.tight_layout()
plt.show()
