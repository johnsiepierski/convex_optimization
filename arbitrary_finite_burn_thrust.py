import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import sys


def create_burn_objective_fn(time, r, v_pre, v_post, min_acceleration, max_acceleration):
    # derive tof boundaries given acceleration constraints

    # construct objective function
    def cost_function(sol):
        if not sol["solution_found"]:
            return sys.float_info.max
        return sol["total_delta_v"]

    def generator(design_variables):

        portion_pre = design_variables[0]
        burn_duration_s = design_variables[1]

        tof_pre_s = portion_pre * burn_duration_s

        tof_post_s = burn_duration_s - tof_pre_s
        r0, v0, r1, v1 = generate_boundary_conditions(r, v_pre, v_post, tof_pre_s, tof_post_s)

        soln = {}
        try:
            # Call your optimization function here
            soln = generate_burn_given_boundary_conditions(r0, v0, r1, v1, burn_duration_s, min_acceleration,
                                                           max_acceleration)
        except Exception as e:
            soln["solution_found"] = False

        return soln

    def objective(design_variables):
        return cost_function(generator(design_variables))

    return objective, generator


def optimize_burn_given_impulse(time, r, v_pre, v_post, min_acceleration, max_acceleration):

    objective, generator = create_burn_objective_fn(time, r, v_pre, v_post, min_acceleration, max_acceleration)

    impulse_delta_v = np.linalg.norm(v_post - v_pre)
    min_duration_s = impulse_delta_v / max_acceleration
    max_duration_s = 500.0

    bounds = [(0, 1), (min_duration_s, max_duration_s)]

    result = differential_evolution(objective, bounds, tol=5.0, popsize=40)
    out = generator(result.x)
    out["result"] = result
    return out




def generate_burn(tof_pre_s, tof_post_s, r, v_pre, v_post):

    # use design variables to generate boundary conditions
    r0, v0, r1, v1 = generate_boundary_conditions(r, v_pre, v_post, tof_pre_s, tof_post_s)

    # optimize trajectory for boundary conditions
    soln = generate_burn_given_boundary_conditions(r0, v0, r1, v1, tof_pre, tof_post)

    return soln




def generate_burn_given_boundary_conditions(r0, v0, r1, v1, T, min_thrust_acceleration, max_thrust_acceleration):

    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24  # mass of the Earth in kg

    # Problem definition
    N = 100  # number of control intervals
    dt = T / N  # length of one control interval

    opti = ca.Opti()

    r = opti.variable(3, N + 1)  # position
    v = opti.variable(3, N + 1)  # velocity
    u = opti.variable(3, N)  # control input (acceleration due to thrust)

    opti.subject_to(r[:, 0] == r0)
    opti.subject_to(v[:, 0] == v0)
    opti.subject_to(r[:, -1] == r1)
    opti.subject_to(v[:, -1] == v1)

    # Dynamics constraints
    for i in range(N):
        r_mid = (r[:, i] + r[:, i + 1]) / 2
        dist_mid = ca.sqrt(ca.dot(r_mid, r_mid))
        g_mid = -G * M / dist_mid ** 3 * r_mid  # gravitational acceleration

        opti.subject_to(r[:, i + 1] == r[:, i] + dt * v[:, i] + 0.5 * dt ** 2 * g_mid)
        opti.subject_to(v[:, i + 1] == v[:, i] + dt * (g_mid + u[:, i]))

    for i in range(N):
        opti.subject_to(ca.sqrt(ca.mtimes(u[:, i].T, u[:, i])) <= max_thrust_acceleration)
        opti.subject_to(ca.sqrt(ca.mtimes(u[:, i].T, u[:, i])) >= min_thrust_acceleration)

    # Objective
    #objective = ca.sum1(ca.norm_2(u)) * dt
    objective = sum(ca.norm_2(u[:, i]) for i in range(N)) * dt

    #objective = ca.sumsqr(u) * dt
    opti.minimize(objective)

    # Initial guess
    opti.set_initial(r, np.linspace(r0, r1, N + 1).T)
    opti.set_initial(v, np.zeros((N + 1, 3)).T)
    opti.set_initial(u, np.ones((N, 3)).T * 5.0)  # Small constant thrust

    # Solve
    opti.solver('ipopt', {'error_on_fail': False})
    sol = opti.solve()

    # Extract solution
    r_opt = sol.value(r)
    v_opt = sol.value(v)
    u_opt = sol.value(u)
    thrust_magnitude = np.linalg.norm(u_opt, axis=0)

    # Calculate total delta-v
    delta_v_total = np.sum(thrust_magnitude) * dt

    out = {}
    out["solution_found"] = sol.stats()["success"]
    out["r"] = r_opt
    out["v"] = v_opt
    out["u"] = u_opt
    out["thrust_magnitude"] = thrust_magnitude
    out["total_delta_v"] = delta_v_total
    out["time"] = np.linspace(0, T, N + 1)
    out["r1"] = r1
    out["v1"] = v1
    return out


def plot_burn_solution(sol):

    r_opt = sol["r"]
    v_opt = sol["v"]
    u_opt = sol["u"]
    thrust_magnitude = sol["thrust_magnitude"]
    time = sol["time"]
    r1 = sol["r1"]
    v1 = sol["v1"]

    plt.figure(figsize=(15, 5))

    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(time, r_opt[0, :], label='x')
    plt.plot(time, r_opt[1, :], label='y')
    plt.plot(time, r_opt[2, :], label='z')
    plt.plot(time[-1], r1[0], 'r*', markersize=10)
    plt.plot(time[-1], r1[1], 'g*', markersize=10)
    plt.plot(time[-1], r1[2], 'b*', markersize=10)
    plt.title('Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    # Velocity plot
    plt.subplot(2, 2, 2)
    plt.plot(time, v_opt[0, :], label='vx')
    plt.plot(time, v_opt[1, :], label='vy')
    plt.plot(time, v_opt[2, :], label='vz')
    plt.plot(time[-1], v1[0], 'r*', markersize=10)
    plt.plot(time[-1], v1[1], 'g*', markersize=10)
    plt.plot(time[-1], v1[2], 'b*', markersize=10)
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    # Thrust plot
    plt.subplot(2, 2, 3)
    plt.step(time[:-1], u_opt[0, :], label='ux')
    plt.step(time[:-1], u_opt[1, :], label='uy')
    plt.step(time[:-1], u_opt[2, :], label='uz')
    plt.title('Thrust Components vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.legend()

    # Thrust magnitude plot
    plt.subplot(2, 2, 4)
    plt.step(time[:-1], thrust_magnitude, label='Thrust Magnitude')
    plt.title('Thrust Magnitude vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust Magnitude (N)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def rk4_step(func, y, t, dt):
    """
    Perform a single Runge-Kutta (RK4) step.

    :param func: The function to integrate. It must be a function of time t and state y.
    :param y: Current state.
    :param t: Current time.
    :param dt: Time step.
    :return: State at time t + dt.
    """
    k1 = dt * func(y, t)
    k2 = dt * func(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * func(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * func(y + k3, t + dt)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def two_body_dynamics(state, t, mu=398600.4418):
    """
    Computes the derivatives of state for the two-body problem.

    :param state: Current state [position, velocity].
    :param t: Time (not used as the two-body problem is time-invariant).
    :param mu: Gravitational parameter (default is for Earth).
    :return: Derivatives of state [velocity, acceleration].
    """
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)

    # Acceleration due to gravity
    a = -mu * r / r_norm**3

    return np.concatenate([v, a])

def propagate_orbit_numerically(initial_position, initial_velocity, time_of_flight, dt=60, mu=398600.4418):
    """
    Propagates an orbit using numerical integration (RK4) under two-body dynamics.

    :param initial_position: Numpy array of initial position in km.
    :param initial_velocity: Numpy array of initial velocity in km/s.
    :param time_of_flight: Time of flight in seconds.
    :param dt: Time step for numerical integration in seconds.
    :param mu: Gravitational parameter in km^3/s^2. Default is for Earth.
    :return: Final position and velocity as numpy arrays.
    """
    # Initial state [position, velocity]
    state = np.concatenate([initial_position, initial_velocity])

    # Time integration
    t = 0
    while abs(t) < abs(time_of_flight):
        state = rk4_step(two_body_dynamics, state, t, dt)
        t += dt

    final_position = state[:3]
    final_velocity = state[3:]

    return final_position, final_velocity

def generate_boundary_conditions(r, v_pre, v_post, tof_pre, tof_post):

    r0, v0 = propagate_orbit_numerically(r, v_pre, -tof_pre, -0.5)
    r1, v1 = propagate_orbit_numerically(r, v_post, tof_post, 0.5)

    return r0, v0, r1, v1




# Initial conditions
#r0 = np.array([6771000.0, 0, 0]) # initial position
#v0 = np.array([0, 7672.4904132836045, 0])  # initial velocity

#rf = np.array([6.20464485e+06, 2.73867844e+06, 0.0])  # final position
#vf = np.array([-3100.85365, 7331.52220, 0.0])  # final velocity

#tof = 360




#generate_burn(r0, v0, r1, v1, tof_pre + tof_post)

# Initial conditions
ri = np.array([6771000.0, 0, 0]) # initial position
vi = np.array([0, 7672.4904132836045, 0])  # initial velocity

rf = np.array([6.20464485e+06, 2.73867844e+06, 0.0])  # final position
vf = np.array([-3100.85365, 7331.52220, 0.0])  # final velocity

#generate_burn(ri, vi, rf, vf, 360.0)



# Calculate total delta-v
#impulse_delta_v_total = np.linalg.norm(delta_v)

# Print total delta-v
#print(f"Impulse delta-v: {impulse_delta_v_total:.2f} m/s")
