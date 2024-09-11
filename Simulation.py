import numpy as np
import matplotlib.pyplot as plt  # Plotting
from matplotlib.patches import Rectangle, Circle
from DifferentialDriveRobot import *


def simulation():
    DT = 0.1
    SIM_TIME = 10.0
    animate = True
    time = 0.0

    robot = DifferentialDriveRobot(0.25, 0.1)

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

    poses = np.copy(robot.get_state())

    while time < SIM_TIME:
        time += DT

        x = robot.step(DT)

        poses = np.hstack((poses, x))

        if animate:
            plt.cla()

            ax.plot(poses[1, :], poses[2, :])

            ax.axis("equal")
            ax.grid(True)
            plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    simulation()
