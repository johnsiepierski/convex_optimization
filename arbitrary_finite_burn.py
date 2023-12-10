import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def generate_burn(r0, v0, r1, v1, tof):

    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24    # mass of the Earth in kg
    R = 6.371e6     # radius of the Earth in m

    # Problem definition
    N = 50  # number of control intervals
    T = tof
    dt = T / N  # length of one control interval

    opti = ca.Opti()

    # Variables (3D)
    r = opti.variable(3, N+1)  # position
    v = opti.variable(3, N+1)  # velocity
    u = opti.variable(3, N)    # control input (acceleration due to thrust)

    opti.subject_to(r[:, 0] == r0)
    opti.subject_to(v[:, 0] == v0)
    opti.subject_to(r[:, -1] == r1)
    opti.subject_to(v[:, -1] == v1)

    # Dynamics constraints (3D)
    for i in range(N):
        r_mid = (r[:, i] + r[:, i+1]) / 2
        dist_mid = ca.sqrt(ca.dot(r_mid, r_mid))
        g_mid = -G * M / dist_mid**3 * r_mid  # gravitational acceleration

        opti.subject_to(r[:, i+1] == r[:, i] + dt*v[:, i] + 0.5 * dt**2 * g_mid)
        opti.subject_to(v[:, i+1] == v[:, i] + dt*(g_mid + u[:, i]))

    # Control constraints
    max_thrust_acceleration = 30.0  # m/s^2
    for i in range(3):
        opti.subject_to(opti.bounded(-max_thrust_acceleration, u[i, :], max_thrust_acceleration))

    # Objective
    objective = ca.sumsqr(u) * dt
    opti.minimize(objective)

    # Initial guess
    opti.set_initial(r, np.linspace(r0, r1, N+1).T)
    opti.set_initial(v, np.zeros((N+1, 3)).T)
    opti.set_initial(u, np.ones((N, 3)).T * 5.0)     # Small constant thrust

    # Solve
    opti.solver('ipopt')
    sol = opti.solve()

    # Extract solution
    r_opt = sol.value(r)
    v_opt = sol.value(v)
    u_opt = sol.value(u)

    # Calculate total delta-v
    delta_v_total = np.sum(np.linalg.norm(u_opt, axis=0)) * dt

    # Print total delta-v
    print(f"Total delta-v: {delta_v_total:.2f} m/s")

    # Plotting
    time = np.linspace(0, T, N+1)
    time_u = np.linspace(0, T-dt, N)

    plt.figure(figsize=(15, 5))

    # Position plot
    plt.subplot(1, 3, 1)
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
    plt.subplot(1, 3, 2)
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
    plt.subplot(1, 3, 3)
    plt.step(time_u, u_opt[0, :], label='ux')
    plt.step(time_u, u_opt[1, :], label='uy')
    plt.step(time_u, u_opt[2, :], label='uz')
    plt.title('Thrust Acceleration vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust Acceleration (m/s²)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def propagate_orbit(initial_position, initial_velocity, time_of_flight, mu=398600.4418):
    """
    Propagates an orbit using two-body motion.

    :param initial_position: Numpy array of initial position in km.
    :param initial_velocity: Numpy array of initial velocity in km/s.
    :param time_of_flight: Time of flight in seconds.
    :param mu: Gravitational parameter in km^3/s^2. Default is for Earth.
    :return: Tuple of final position and final velocity as numpy arrays.
    """
    # Convert position and velocity to radial and tangential components
    r0 = np.linalg.norm(initial_position)
    v0 = np.linalg.norm(initial_velocity)
    vr0 = np.dot(initial_velocity, initial_position) / r0

    # Specific angular momentum
    h = np.cross(initial_position, initial_velocity)
    h_norm = np.linalg.norm(h)

    # Semi-major axis and eccentricity
    a = 1 / ((2 / r0) - (v0 ** 2 / mu))
    e = np.sqrt(1 - (h_norm ** 2) / (a * mu))

    # True anomaly
    cos_f0 = (a * (1 - e ** 2) / r0 - 1) / e
    sin_f0 = (a * (1 - e ** 2) * vr0) / (e * h_norm)
    f0 = np.arctan2(sin_f0, cos_f0)

    # Mean anomaly
    E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f0 / 2))
    M0 = E0 - e * np.sin(E0)

    # Mean motion
    n = np.sqrt(mu / a ** 3)

    # Propagate mean anomaly
    M = M0 + n * time_of_flight

    # Solve Kepler's equation for E
    E = M
    for _ in range(10):  # Newton-Raphson iteration
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    # True anomaly
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # Final position and velocity
    r = a * (1 - e ** 2) / (1 + e * np.cos(f))
    final_position = r * (np.cos(f) * initial_position / r0 + np.sin(f) * np.cross(h, initial_position) / h_norm)
    final_velocity = np.cross(h, final_position) / r - (mu / h_norm) * final_position / r

    return final_position, final_velocity

def generate_boundary_conditions(r, v_pre, v_post, tof_pre, tof_post):

    r0, v0 = propagate_orbit(r, v_pre, -tof_pre)
    r1, v1 = propagate_orbit(r, v_post, tof_post)

    return r0, v0, r1, v1




# Initial conditions (3D)
r0 = np.array([6771000.0, 0, 0]) # initial position
v0 = np.array([0, 7672.4904132836045, 0])  # initial velocity

rf = np.array([6.20464485e+06, 2.73867844e+06, 0.0])  # final position
vf = np.array([-3100.85365, 7331.52220, 0.0])  # final velocity

tof = 360

generate_burn(r0,v0, rf, vf, tof)