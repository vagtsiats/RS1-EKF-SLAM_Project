import numpy as np
import matplotlib.pyplot as plt  # Plotting
from matplotlib.patches import Rectangle, Circle
from DifferentialDriveRobot import *
import slam
from definitions import *


def simulation():
    time = 0.0

    lm_map = hp.generate_random_points(50, [-5, 5], [-5, 5])
    targets = hp.generate_random_points(10, [-5, 5], [-5, 5])
    cur_targ = 0

    robot = DifferentialDriveRobot(1, 0.5)
    gt_path = np.zeros((3, 1))  # ground truth pose

    miu_init = np.zeros((slam.STATE_SIZE, 1))
    sigma_init = np.eye(slam.STATE_SIZE) * 0.0001

    slam_estimation = [(miu_init, sigma_init)]

    print("Simulation")
    while time < SIM_TIME:

        estimated_pose = slam_estimation[-1][0][:3].reshape((-1, 1))

        if np.linalg.norm(estimated_pose[:2, 0] - targets[cur_targ]) < 0.5:
            cur_targ += 1
            cur_targ = min(len(targets) - 1, cur_targ)

        u = robot.compute_control(estimated_pose, targets[cur_targ])

        robot.step(control=u, dt=DT)

        u_bad = np.copy(u) + np.random.normal(0, MOTION_ERROR)
        gt_path = np.hstack(
            (
                gt_path,
                hp.RK4(robot.forwardKinematics, gt_path[:, -1].reshape(-1, 1), u_bad, DT),
            )
        )

        lidar_z = robot.get_lidar_measurements(gt_path[:, -1].reshape(-1, 1), lm_map)
        odo_u = robot.get_controls_from_odometry()

        miu, sigma = slam_estimation[-1]
        slam_estimation.append(slam.ekf_slam_step(miu, sigma, odo_u, lidar_z, R, Q))

        if slam.calc_N(miu) > 60:
            break

        time += DT

    odo_path = robot.get_odometry()
    est_path = np.array([t[0][:3, 0].tolist() for t in slam_estimation]).T

    fig1, ax1 = plt.subplots()

    time_steps = np.arange(len(est_path.T))

    ax1.plot(time_steps, np.linalg.norm(gt_path - odo_path, axis=0))
    ax1.plot(time_steps, np.linalg.norm(gt_path - est_path, axis=0))
    plt.show()

    if animation:
        print("Animation")
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

        # Draw static landmarks once
        for lm in lm_map:
            landmark = Circle(lm, 0.01, color="black", fill="True")
            ax.add_patch(landmark)

        # Prepare lines for real-time updates
        (gt_line,) = ax.plot([], [], label="Ground Truth", color="blue")
        (odo_line,) = ax.plot([], [], label="Odometry", color="orange")
        (est_line,) = ax.plot([], [], label="Estimated Path", color="red")

        slam_circle = Circle((gt_path[0, 0], gt_path[1, 0]), 0.1)

        ax.add_patch(slam_circle)

        ellipses = []

        # Set static properties of the plot
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)
        ax.legend()

        for t in range(len(odo_path[0, :])):
            # Update ground truth and odometry lines instead of replotting
            # plt.cla()
            gt_line.set_data(gt_path[0, :t], gt_path[1, :t])
            odo_line.set_data(odo_path[0, :t], odo_path[1, :t])
            est_line.set_data(est_path[0, :t], est_path[1, :t])

            # hp.draw_ellipse_with_cross(ax, slam_estimation[t][0][:2].T, slam_estimation[t][1][:2, :2])

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

            # slam_circle.set_center((gt_poses[0, t], gt_poses[1, t]))

            plt.pause(DT)

        plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=200, suppress="True")
    simulation()
