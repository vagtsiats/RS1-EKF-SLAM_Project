import numpy as np
import helpers as hp


class DifferentialDriveRobot:
    odometry_poses = np.zeros((3, 1))  # [x,y,theta]

    def __init__(self, d, r) -> None:
        self.d = d
        self.r = r

        self.lidar_width = np.pi / 2
        self.lidar_dist = 10
        self.lidar_noise = 1e-3

    def forwardKinematics_integration(self, pose, control, dt=0.1):

        kin = np.array(
            [
                [(self.r / 2) * np.cos(pose[2][0]), (self.r / 2) * np.cos(pose[2][0])],
                [(self.r / 2) * np.sin(pose[2][0]), (self.r / 2) * np.sin(pose[2][0])],
                [-self.r / (2 * self.d), self.r / (2 * self.d)],
            ]
        )

        return pose + kin @ control * dt

    # used for odometry
    def bodyTwist_integration(self, pose, control, dt):
        H_odom = np.array(
            [
                [self.r / 2.0, self.r / 2.0],
                [0.0, 0.0],
                [-self.r / (2.0 * self.d), self.r / (2.0 * self.d)],
            ]
        )
        Vb = H_odom @ control

        R = np.array(
            [
                [np.cos(pose[2, 0]), -np.sin(pose[2, 0]), 0.0],
                [np.sin(pose[2, 0]), np.cos(pose[2, 0]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        if abs(Vb[2, 0]) < 1e-6:
            return pose + R @ Vb * dt
        else:
            vx = Vb[0, 0]
            vy = Vb[1, 0]
            omega = Vb[2, 0]
            dp = np.array(
                [
                    [(vx * np.sin(omega) + vy * (np.cos(omega) - 1.0)) / omega],
                    [(vy * np.sin(omega) + vx * (1.0 - np.cos(omega))) / omega],
                    [omega],
                ]
            )
            return pose + R @ dp * dt

    def __trim_control(self, control):
        return np.clip(control, -5, 5)

    # control = [uL, uR]
    def step(self, control, dt=0.1):

        self.odometry_poses = np.hstack(
            (
                self.odometry_poses,
                self.bodyTwist_integration(
                    self.odometry_poses[:, -1].reshape(-1, 1), control=control, dt=dt
                ),
            )
        )

    def get_odometry(self):
        return self.odometry_poses

    def get_controls_from_odometry(self):
        dx = (
            self.odometry_poses[:, -1].reshape(-1, 1)[0, 0]
            - self.odometry_poses[:, -2].reshape(-1, 1)[0, 0]
        )
        dy = (
            self.odometry_poses[:, -1].reshape(-1, 1)[1, 0]
            - self.odometry_poses[:, -2].reshape(-1, 1)[1, 0]
        )
        dtheta = hp.angle_dist(
            self.odometry_poses[:, -1].reshape(-1, 1)[2, 0],
            self.odometry_poses[:, -2].reshape(-1, 1)[2, 0],
        )
        return np.array([[dx, dy, dtheta]]).T

    def get_lidar_measurements(self, pose, lm_map):

        detects = []

        theta = pose[2, 0]  # heading of the robot
        for k in range(len(lm_map)):
            lx = lm_map[k][0] + np.random.randn() * self.lidar_noise
            ly = lm_map[k][1] + np.random.randn() * self.lidar_noise
            xx = lx - pose[1, 0]
            yy = ly - pose[2, 0]
            rel = np.arctan2(yy, xx)
            rel = hp.angle_dist(rel, theta)
            r = np.sqrt(xx**2 + yy**2)

            if (
                rel >= -self.lidar_width / 2.0
                and rel <= self.lidar_width / 2.0
                and r <= self.lidar_dist
            ):

                detects += [np.array([[r, rel]])]  # (distance, angle difference)
        return detects
