import numpy as np

SEED = 3452

# SIM
DT = 0.1
SIM_TIME = 10.0
animation = True

# SLAM
R = np.eye(3) * 0.2  # noise for motion model!
Q = np.eye(2) * 1e-3  # noise for observation model!

MOTION_ERROR = 0.01
LIDAR_NOISE = 1e-3

MH_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,th]
LM_SIZE = 2  # LM state size [x,y]


LIDAR_WIDTH = 2 * np.pi
LIDAR_DIST = 5
