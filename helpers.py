import numpy as np
from matplotlib.patches import Rectangle, Circle, Ellipse


def angle_dist(b, a):
    theta = b - a
    while theta < -np.pi:
        theta += 2.0 * np.pi
    while theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def draw_ellipse_with_cross(ax, mean, covariance):
    mean = mean.reshape((2,))

    # Eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Get the angle of the ellipse (in degrees)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and height of the ellipse are proportional to the square root of eigenvalues
    width, height = 2 * np.sqrt(eigenvalues)

    # Create an ellipse patch
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle, edgecolor="black", fc="None", lw=2
    )
    ax.add_patch(ellipse)

    # Add a small "x" aligned with the ellipse's axes
    cross_length = 0.5  # Fixed length for the "x"
    for eigenvector in eigenvectors.T:
        line_start = mean - cross_length * eigenvector
        line_end = mean + cross_length * eigenvector
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color="black", lw=0.5)
