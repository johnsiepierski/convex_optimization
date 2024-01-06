from arbitrary_finite_burn_thrust import *
import numpy as np


def test_generate_burn_given_boundary_conditions():

    ri = np.array([6771000.0, 0, 0]) # initial position
    vi = np.array([0, 7672.4904132836045, 0])  # initial velocity

    rf = np.array([6.20464485e+06, 2.73867844e+06, 0.0])  # final position
    vf = np.array([-3100.85365, 7331.52220, 0.0])  # final velocity

    max_thrust_acceleration = 30.0  # m/s^2
    min_thrust_acceleration = 0.0  # m/s^2

    soln = generate_burn_given_boundary_conditions(ri, vi, rf, vf, 360.0, min_thrust_acceleration, max_thrust_acceleration)
    total_delta_v = soln["total_delta_v"]
    print(f"Total delta-v: {total_delta_v:.2f} m/s")

    assert total_delta_v < 305.0
    assert total_delta_v > 300.0

def test_optimize_burn_given_impulse():
    # Inputs
    r_impulse = np.array([6771000.0, 0, 0])  # initial position
    v_pre = np.array([0, 7672.4904132836045, 0])  # initial velocity
    delta_v = np.array([100.0, 200.0, 50.0])  # change in velocity
    v_post = v_pre + delta_v

    min_acceleration = 0.0
    max_acceleration = 40.0

    # Calculate total delta-v
    impulse_delta_v_total = np.linalg.norm(delta_v)

    # compute trajectory
    #optimize_burn_given_impulse(time, r, v_pre, v_post, min_acceleration, max_acceleration)

    soln = optimize_burn_given_impulse(0.0, r_impulse, v_pre, v_post, min_acceleration, max_acceleration)
    total_delta_v = soln["total_delta_v"]

    # Print total delta-v
    print(f"Impulse delta-v: {impulse_delta_v_total:.2f} m/s")
    print(f"Finite delta-v: {total_delta_v:.2f} m/s")

    print(soln["result"])

    plot_burn_solution(soln)


def plot_objective_fn():

    # Inputs
    r_impulse = np.array([6771000.0, 0, 0])  # initial position
    v_pre = np.array([0, 7672.4904132836045, 0])  # initial velocity
    delta_v = np.array([100.0, 200.0, 50.0])  # change in velocity
    v_post = v_pre + delta_v

    min_acceleration = 0.0
    max_acceleration = 3000.0

    # Calculate total delta-v
    impulse_delta_v_total = np.linalg.norm(delta_v)

    # Generate a grid of x and y values
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 50.0, 5)
    x, y = np.meshgrid(x, y)

    objective, generator = create_burn_objective_fn(0.0, r_impulse, v_pre, v_post, min_acceleration, max_acceleration)

    # Evaluate the function at each point on the grid
    z = np.array([objective([xi, yi]) for xi, yi in zip(x.flatten(), y.flatten())]).reshape(x.shape)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis', aspect='auto')
    #plt.colorbar(label='Objective function value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Heatmap of the burn Function')
    plt.show()

plot_objective_fn()
#test_optimize_burn_given_impulse()
#test_generate_burn_given_boundary_conditions()