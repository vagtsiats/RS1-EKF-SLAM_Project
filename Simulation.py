import numpy as np
import matplotlib.pyplot as plt  # Plotting
from matplotlib.patches import Rectangle, Circle
from DifferentialDriveRobot import *
import slam


DT = 0.1
SIM_TIME = 100.0
animation = True


def simulation():
    time = 0.0

    lm_map = [
        (4.0, 4.0),
        (4.0, 0.0),
        (4.0, -4.0),
        (0.0, -4.0),
        (-4.0, -4.0),
        (-4.0, 0.0),
        (-4.0, 4.0),
        (0.0, 4.0),
    ]

    robot = DifferentialDriveRobot(1, 0.5)
    gt_poses = np.zeros((3, 1))  # ground truth pose

    R = np.eye(3) * 0.2  # noise for motion model!
    Q = np.eye(2) * 0.001  # noise for observation model!

    miu_init = np.zeros((slam.STATE_SIZE, 1))
    sigma_init = np.eye(slam.STATE_SIZE) * 0.0001

    slam_estimation = [(miu_init, sigma_init)]

    while time < SIM_TIME:
        time += DT

        if time < 1.0:
            u = np.array([[2, 2]]).T
        # elif time > 2.0:
        #     u = np.array([[2, 2]]).T
        else:
            u = np.array([[1, 2]]).T

        robot.step(control=u, dt=DT)

        u_error = 0.3
        u_bad = np.copy(u) * np.random.uniform(low=1 - u_error, high=1 + u_error, size=(2, 1))
        gt_poses = np.hstack(
            (
                gt_poses,
                robot.forwardKinematics_integration(gt_poses[:, -1].reshape(-1, 1), u_bad, DT),
            )
        )

        lidar_z = robot.get_lidar_measurements(gt_poses[:, -1].reshape(-1, 1), lm_map)
        # for z in lidar_z:
        #     print(gt_poses[:, -1][0] + z[0, 0] * np.cos(gt_poses[:, -1][2] + z[0, 1]), gt_poses[:, -1][1] + z[0, 0] * np.sin(gt_poses[:, -1][2] + z[0, 1]))

        odo_u = robot.get_controls_from_odometry()

        miu, sigma = slam_estimation[-1]

        slam_estimation.append(slam.ekf_slam_step(miu, sigma, odo_u, lidar_z, R, Q))

    # print(slam_estimation[10][0].T)
    # print(slam_estimation[10][0][3:5].T)

    # print(slam_estimation[10][1][3:5, 3:5])

    if animation:
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

        odo_poses = robot.get_odometry()

        for t in range(len(odo_poses[0, :]) - 10):

            plt.cla()

            for lm in lm_map:
                landmark = Circle(lm, 0.1, color="black", fill="True")
                ax.add_patch(landmark)

            ax.plot(gt_poses[0, :t], gt_poses[1, :t], label="Ground Truth")
            ax.plot(odo_poses[0, :t], odo_poses[1, :t], label="Odometry")

            hp.draw_ellipse_with_cross(ax, slam_estimation[t][0][:2].T, slam_estimation[t][1][:2, :2])

            N = slam.calc_N(slam_estimation[t][0])
            for lm_id in range(N):
                hp.draw_ellipse_with_cross(
                    ax,
                    slam_estimation[t][0][slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE].T,
                    slam_estimation[t][1][
                        slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE,
                        slam.STATE_SIZE + lm_id * slam.LM_SIZE : slam.STATE_SIZE + (lm_id + 1) * slam.LM_SIZE,
                    ],
                )

            ax.add_patch(
                Circle(
                    slam_estimation[t][0][:2].reshape(
                        2,
                    ),
                    0.1,
                    fill="False",
                )
            )

            ax.legend()
            ax.plot()

            # ax.axis("equal")
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.grid(True)
            plt.pause(DT / 100)

        plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=200, suppress="True")
    simulation()
