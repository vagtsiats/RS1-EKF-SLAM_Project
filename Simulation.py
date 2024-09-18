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

    R = np.eye(3) * 0.1  # noise for motion model!
    Q = np.eye(2) * 0.001  # noise for observation model!

    ekf_slam_miu = np.zeros((slam.STATE_SIZE, 1))
    ekf_slam_sigma = np.eye(slam.STATE_SIZE) * 0.0001

    slam_estimation = [(ekf_slam_miu, ekf_slam_sigma)]

    while time < SIM_TIME:
        time += DT

        if time < 1.0:
            u = np.array([[2, 2]]).T
        # elif time > 2.0:
        #     u = np.array([[2, 2]]).T
        else:
            u = np.array([[2, 4]]).T

        robot.step(control=u, dt=DT)

        u_error = 0.5
        u_bad = np.copy(u) * np.random.uniform(low=1 - u_error, high=1 + u_error, size=(2, 1))
        gt_poses = np.hstack(
            (
                gt_poses,
                robot.forwardKinematics_integration(gt_poses[:, -1].reshape(-1, 1), u_bad, DT),
            )
        )

        lidar_z = robot.get_lidar_measurements(gt_poses[:, -1].reshape(-1, 1), lm_map)
        odo_u = robot.get_controls_from_odometry()

        ekf_slam_miu, ekf_slam_sigma = slam.ekf_slam_step(
            ekf_slam_miu, ekf_slam_sigma, odo_u, lidar_z, R, Q
        )

        slam_estimation.append((ekf_slam_miu, ekf_slam_sigma))

    # print(slam_estimation[0][0][:2].T)

    # print(slam_estimation[0][1])

    if animation:
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect(
            "key_release_event", lambda event: [exit(0) if event.key == "escape" else None]
        )

        odo_poses = robot.get_odometry()

        for t in range(len(odo_poses[0, :]) - 10):

            plt.cla()

            for lm in lm_map:
                landmark = Circle(lm, 0.1, color="black", fill="True")
                ax.add_patch(landmark)

            ax.plot(gt_poses[0, :t], gt_poses[1, :t], label="Ground Truth")
            ax.plot(odo_poses[0, :t], odo_poses[1, :t], label="Odometry")

            # hp.draw_ellipse_with_cross(
            #     ax, slam_estimation[t][0][:2].T, slam_estimation[t][1][:2, :2]
            # )

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

            ax.axis("equal")
            ax.grid(True)
            plt.pause(DT / 100)

        plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=200)
    simulation()
