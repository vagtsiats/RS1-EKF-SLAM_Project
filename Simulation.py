import numpy as np
import matplotlib.pyplot as plt  # Plotting
from matplotlib.patches import Rectangle, Circle
from DifferentialDriveRobot import *
import slam
from definitions import *


def simulation():
    time = 0.0

    lm_map = hp.generate_random_points(50, [-10, 10], [-10, 10])
    targets = hp.generate_random_points(10, [-10, 10], [-10, 10])
    lm_map = [(-5, 5), (5, 5), (0, 5), (-5, -5), (5, -5), (0, -5)]
    targets = [(7, 7), (7, 0), (7, 0), (-7, 0)]

    cur_targ = 0

    gt_path = np.array([[-7, 7, 0]]).T  # ground truth initial pose
    robot = DifferentialDriveRobot(1, 0.5, gt_path)

    miu_init = np.copy(gt_path)
    sigma_init = np.eye(slam.STATE_SIZE) * 0.0001

    slam_estimation = [(miu_init, sigma_init)]

    print("Simulation")
    while time < SIM_TIME:

        estimated_pose = slam_estimation[-1][0][:3].reshape((-1, 1))

        if np.linalg.norm(estimated_pose[:2, 0] - targets[cur_targ]) < 0.5:
            cur_targ += 1
            cur_targ = min(len(targets) - 1, cur_targ)
            if np.linalg.norm(estimated_pose[:2, 0] - targets[-1]) < 0.5:
                break

        u = robot.compute_control(estimated_pose, targets[cur_targ])
        u_real = np.copy(u)
        u_real[1, 0] = 0.9 * u_real[1, 0]

        gt = hp.RK4(robot.forwardKinematics, gt_path[:, -1].reshape(-1, 1), u_real, DT)  # + np.random.normal(0, MOTION_ERROR, (3, 1))
        gt_path = np.hstack((gt_path, gt))

        u_bad = np.copy(u_real) + np.random.normal(0, ENCODER_NOISE, (2, 1))
        robot.odometry(u=u_bad, dt=DT)

        lidar_z = robot.get_lidar_measurements(gt_path[:, -1].reshape(-1, 1), lm_map)
        odo_u = robot.get_controls_from_odometry()

        miu, sigma = slam_estimation[-1]
        slam_estimation.append(slam.ekf_slam_step(miu, sigma, odo_u, lidar_z, R, Q))

        if slam.calc_N(miu) > 150:
            break

        time += DT

    odo_path = robot.get_odometry()
    est_path = np.array([t[0][:3, 0].tolist() for t in slam_estimation]).T

    fig, ax = plt.subplots()

    time_steps = np.arange(len(est_path.T)) * DT

    ax.plot(time_steps, np.linalg.norm(gt_path - odo_path, axis=0), label="Odometry Error", color="red")
    ax.plot(time_steps, np.linalg.norm(gt_path - est_path, axis=0), label="Estimation Error", color="orange")
    ax.legend()
    ax.set_xlabel("Time(sec)")
    ax.set_ylabel("State Error Norm")
    # plt.show()

    if animation:
        print("Animation")
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

        # Draw static landmarks once
        for lm in lm_map:
            landmark = Circle(lm, 0.1, color="black", fill="True")
            ax.add_patch(landmark)

        # Prepare lines for real-time updates
        (gt_line,) = ax.plot([], [], label="Ground Truth", color="blue")
        (odo_line,) = ax.plot([], [], label="Odometry", color="red")
        (est_line,) = ax.plot([], [], label="Estimated Path", color="orange")

        ellipses = []

        gt_circle = Circle(gt_path[:, 0], 0.15, fill="False", color="blue")

        ax.add_patch(gt_circle)

        p_ellipse, p_lines = hp.ellipse_with_cross(ax, slam_estimation[0][0][:2].T, slam_estimation[0][1][:2, :2])

        # Set static properties of the plot
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.grid(True)
        ax.legend()

        for t in range(len(odo_path[0, :])):
            # Update ground truth and odometry lines instead of replotting
            # plt.cla()
            gt_line.set_data(gt_path[0, : t + 1], gt_path[1, : t + 1])
            odo_line.set_data(odo_path[0, : t + 1], odo_path[1, : t + 1])
            est_line.set_data(est_path[0, : t + 1], est_path[1, : t + 1])

            gt_circle.set_center(gt_path[:, t])

            hp.ellipse_with_cross(ax, slam_estimation[t][0][:2].T, slam_estimation[t][1][:2, :2], p_ellipse, p_lines)

            N = slam.calc_N(slam_estimation[t][0])
            for lm_id in range(N):
                mean_id = slam_estimation[t][0][slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE].T
                cov_id = slam_estimation[t][1][
                    slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE,
                    slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE,
                ]

                if 0 <= lm_id < len(ellipses):
                    hp.ellipse_with_cross(ax, mean_id, cov_id, ellipses[lm_id][0], ellipses[lm_id][1])
                else:
                    ellipses.append(hp.ellipse_with_cross(ax, mean_id, cov_id))

            plt.pause(DT)

        plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=200, suppress="True")
    simulation()
