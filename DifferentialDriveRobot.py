import numpy as np
import helpers as hp
from definitions import *


class DifferentialDriveRobot:
    odometry_poses = np.zeros((3, 1))  # [x,y,theta]

    def __init__(self, d, r) -> None:
        self.wheel_base = d
        self.wheel_radius = r

    def forwardKinematics(self, pose, control):

        kin = np.array(
            [
                [(self.wheel_radius / 2) * np.cos(pose[2][0]), (self.wheel_radius / 2) * np.cos(pose[2][0])],
                [(self.wheel_radius / 2) * np.sin(pose[2][0]), (self.wheel_radius / 2) * np.sin(pose[2][0])],
                [-self.wheel_radius / (2 * self.wheel_base), self.wheel_radius / (2 * self.wheel_base)],
            ]
        )

        return kin @ control

    # used for odometry
    def bodyTwist(self, pose, control):
        H_odom = np.array(
            [
                [self.wheel_radius / 2.0, self.wheel_radius / 2.0],
                [0.0, 0.0],
                [-self.wheel_radius / (2.0 * self.wheel_base), self.wheel_radius / (2.0 * self.wheel_base)],
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
            return R @ Vb
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
            return R @ dp

    def __trim_control(self, control):

        # Find the largest ratio (if greater than 1, scale)
        max_ratio = np.max(abs(control) / 10)
        if max_ratio > 1:
            control = control / max_ratio

        return control

    def compute_control(self, current_pose, target):
        target = np.array(target).reshape((-1, 1))

        # Compute the error in position
        d = target - current_pose[:2, 0].reshape((-1, 1))

        # Distance to the target
        distance_error = (d.T @ d)[0, 0]
        orientation_error = hp.angle_dist(np.arctan2(d[1, 0], d[0, 0]), current_pose[2, 0])

        v = np.clip(distance_error, -1, 1) * 3
        w = np.clip(orientation_error, -np.pi, np.pi) * 10

        # Compute wheel angular velocities
        control = np.zeros((2, 1))
        control[0, 0] = (v - w * self.wheel_base / 2) / self.wheel_radius
        control[1, 0] = (v + w * self.wheel_base / 2) / self.wheel_radius

        return self.__trim_control(control)

    # control = [uL, uR]
    def step(self, control, dt=0.1):

        self.odometry_poses = np.hstack(
            (
                self.odometry_poses,
                hp.RK4(self.bodyTwist, self.odometry_poses[:, -1].reshape(-1, 1), u=control, dt=dt),
            )
        )

    def get_odometry(self):
        return self.odometry_poses

    def get_controls_from_odometry(self):
        dx = self.odometry_poses[:, -1].reshape(-1, 1)[0, 0] - self.odometry_poses[:, -2].reshape(-1, 1)[0, 0]
        dy = self.odometry_poses[:, -1].reshape(-1, 1)[1, 0] - self.odometry_poses[:, -2].reshape(-1, 1)[1, 0]
        dtheta = hp.angle_dist(
            self.odometry_poses[:, -1].reshape(-1, 1)[2, 0],
            self.odometry_poses[:, -2].reshape(-1, 1)[2, 0],
        )
        return np.array([[dx, dy, dtheta]]).T

    def get_lidar_measurements(self, pose, lm_map):

        detects = []

        theta = pose[2, 0]  # heading of the robot
        for lm in lm_map:
            lx = lm[0] + np.random.normal(0, LIDAR_NOISE)
            ly = lm[1] + np.random.normal(0, LIDAR_NOISE)

            xx = lx - pose[0, 0]
            yy = ly - pose[1, 0]
            rel = np.arctan2(yy, xx)
            rel = hp.angle_dist(rel, theta)
            r = np.sqrt(xx**2 + yy**2)

            if rel >= -LIDAR_WIDTH / 2.0 and rel <= LIDAR_WIDTH / 2.0 and r <= LIDAR_DIST:

                detects += [np.array([[r, rel]])]  # (distance, angle difference)
        return detects
