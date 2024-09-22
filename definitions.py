import numpy as np


SEED = 24383

# SIM
DT = 0.1
SIM_TIME = 30.0
animation = True

# SLAM
R = np.eye(3) * 0.1  # noise for motion model!
Q = np.eye(2) * 0.01  # noise for observation model!


ENCODER_NOISE = 0.1
LIDAR_NOISE = 1e-3

MH_DIST_TH = 2  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,th]
LM_SIZE = 2  # LM state size [x,y]


LIDAR_WIDTH = np.pi
LIDAR_DIST = 10
