import numpy as np


# f  = motion model
# Jf = motion model derivative
# h  = observation model
# Jh = observation model derivative


def ekf_slam(_miu, _sigma, _u, _z):
    # prediction
    miu_bar = f(_miu, _u)
    sigma_bar = Jf @ _sigma @ Jf.T + R

    # correction
    K = sigma_bar @ Jh.T @ np.inv(Jh @ sigma_bar @ Jh.T + Q)

    miu = miu_bar + K @ (_z - h(miu_bar))
    sigma = (np.eye(miu.shape()[0]) - K @ Jh) @ sigma_bar

    return miu, sigma
