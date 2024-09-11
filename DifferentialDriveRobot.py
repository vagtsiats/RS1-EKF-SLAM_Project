import numpy as np


class DifferentialDriveRobot:
    state = np.zeros((3, 1))  # [x,y,theta]
    thd_L = 0
    thd_R = 0
    axes = None
    prevx = np.empty((3, 1))

    def __init__(self, d, r, state=None) -> None:
        self.d = d
        self.r = r

        if state:
            self.prevx = np.hstack((self.prevx, state.reshape(-1, 1)))
            self.state = state
        else:
            self.prevx = np.hstack((self.prevx, self.state.reshape(-1, 1)))

    # forward differential kinematics function
    def diffkin(self, control):
        th = self.state[0][0]
        r_2 = self.r / 2
        d = self.d

        control.reshape(1, -1)

        mat = np.array([[-r_2 / d, r_2 / d], [r_2 * np.cos(th), r_2 * np.cos(th)], [r_2 * np.sin(th), r_2 * np.sin(th)]])

        c = self.trim_control(control)

        return (mat @ c).reshape(-1, 1)

    def odometry(self):
        pass

    def visualize(self, ax=None):
        if self.axes == None:
            # self.fig, self.ax = plt.
            pass
        pass

    def trim_control(self, control):
        return np.clip(control, -5, 5)

    def step(self, dt=0.1):
        dx = self.diffkin(np.array([0.1, 1]))

        self.state += dx * dt

        self.prevx = np.hstack((self.prevx, self.state.reshape(-1, 1)))

        return self.state

    def get_state(self):
        return self.state
