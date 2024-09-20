import numpy as np
from matplotlib.patches import Rectangle, Circle, Ellipse
from definitions import SEED


def RK4(diffkinfunc, x, u, dt=0.1):

    f1 = diffkinfunc(x, u)
    f2 = diffkinfunc(x + f1 * dt / 2, u)
    f3 = diffkinfunc(x + f2 * dt / 2, u)
    f4 = diffkinfunc(x + f3 * dt, u)

    return x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)


def angle_dist(b, a):
    theta = b - a
    while theta < -np.pi:
        theta += 2.0 * np.pi
    while theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def generate_random_points(n, xlim, ylim):

    np.random.seed(SEED)

    x_random = np.random.uniform(xlim[0], xlim[1], n)
    y_random = np.random.uniform(ylim[0], ylim[1], n)

    return list(zip(x_random, y_random))


def ellipse_with_cross(ax, mean, covariance, ellipse=None, lines=None):
    mean = mean.reshape((2,))

    # Eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Get the angle of the ellipse (in degrees)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Width and height of the ellipse are proportional to the square root of eigenvalues
    width, height = 2 * np.sqrt(eigenvalues)

    if ellipse is None or lines is None:
        # Create ellipse and cross if they do not exist yet (initialization)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor="black", fc="None", lw=1)
        ax.add_patch(ellipse)

        cross_length = 0.2  # Fixed length for the "x"
        (line1,) = ax.plot([], [], color="black", lw=0.5)
        (line2,) = ax.plot([], [], color="black", lw=0.5)
        lines = [line1, line2]
    else:
        # Update existing ellipse
        ellipse.set_center(mean)
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle

        # Update the cross lines
        cross_length = 0.2
        line1_start = mean - cross_length * eigenvectors[:, 0]
        line1_end = mean + cross_length * eigenvectors[:, 0]
        lines[0].set_data([line1_start[0], line1_end[0]], [line1_start[1], line1_end[1]])

        line2_start = mean - cross_length * eigenvectors[:, 1]
        line2_end = mean + cross_length * eigenvectors[:, 1]
        lines[1].set_data([line2_start[0], line2_end[0]], [line2_start[1], line2_end[1]])

    return ellipse, lines
