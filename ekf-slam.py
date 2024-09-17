import numpy as np

# miu : [x, y, th, lm1x, lm1y, lm2x, lm2y, ... ].T
# sigma :
# f  : motion model
# Jf : motion model derivative
# h  : observation model
# Jh : observation model derivative
# u  : controls = [v, w].T
# z  : observations = real observations from sensors [x, y, signature] why signature???
# R  : motion noise
# Q  : observation noise

MH_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,th]
LM_SIZE = 2  # LM state size [x,y]


def ekf_slam_step(miu_, sigma_, u_, z_, R_, Q_):

    N_ = int((miu_.shape[0] - STATE_SIZE) / LM_SIZE)

    # prediction
    Fx = np.hstack((np.eye(3), np.zeros((3, LM_SIZE * N_))))

    miu_bar = miu_ + Fx.T @ f(miu_[:, 0], u_)

    Gt = np.eye(3) + Fx.T @ Jf(miu_[:, 0], u_) @ Fx

    sigma_bar = Gt @ sigma_ @ Gt.T + Fx.T @ R_ @ Fx

    # correction
    for i in range(z_.shape[0]):

        new_landmark_miu = np.zeros((2, 1))
        new_landmark_miu[0, 0] = miu_bar[0, 0] + z_[i, 0] * np.cos(miu_bar[2, 0] + z_[i, 1])
        new_landmark_miu[1, 0] = miu_bar[1, 0] + z_[i, 0] * np.sin(miu_bar[2, 0] + z_[i, 1])

        mah_dist = np.zeros((1, N_ + 1))
        for k in range(N_):
            z_expected_k = h(miu_bar[:STATE_SIZE, 0], miu_bar[STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE, 0])

            Fxk = np.zeros(STATE_SIZE + LM_SIZE, LM_SIZE * N_ + STATE_SIZE)
            Fxk[:STATE_SIZE, :STATE_SIZE] = np.eye(STATE_SIZE)
            Fxk[STATE_SIZE:, STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE] = np.eye(LM_SIZE)

            Hk = H(miu_bar[:STATE_SIZE, 0], miu_bar[STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE, 0]) @ Fxk

            psi_k = Hk @ sigma_bar @ Hk.T + Q_

            # Mahalanobis Distance
            mah_dist[k] = (z_[i, :].T - z_expected_k).T @ np.linalg.pinv(psi_k) @ (z_[i, :].T - z_expected_k)

        mah_dist[N_] = MH_DIST_TH

        j = np.argmin(mah_dist)  # ML correspondence selection

        N_ = np.max(N_, j)
        # create new landmark
        if N_ == j:
            miu_bar = np.vstack(miu_bar, new_landmark_miu)

            sigma_bar = np.vstack(
                (
                    np.hstack((sigma_bar, np.zeros((sigma_bar.shape[0], LM_SIZE)))),
                    np.hstack((np.zeros((LM_SIZE, sigma_bar.shape[1])), np.eye(LM_SIZE))),
                )
            )

        Fxj = np.zeros(STATE_SIZE + LM_SIZE, LM_SIZE * N_ + STATE_SIZE)
        Fxj[:STATE_SIZE, :STATE_SIZE] = np.eye(STATE_SIZE)
        Fxj[STATE_SIZE:, STATE_SIZE + j * LM_SIZE : STATE_SIZE + (j + 1) * LM_SIZE] = np.eye(LM_SIZE)

        Hj = H(miu_bar[:STATE_SIZE, 0], miu_bar[STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE, 0]) @ Fxj
        psi_j = Hj @ sigma_bar @ Hj.T + Q_

        K = sigma_bar @ Hj.T @ np.linalg.pinv(psi_j)

        z_expected_j = h(miu_bar[:STATE_SIZE, 0], miu_bar[STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE, 0])

        miu_bar = miu_bar + K @ (z_[i, :] - z_expected_j)
        sigma_bar = (np.eye(np.shape(sigma_bar)) - K @ Hj) @ sigma_bar

    return miu_bar, sigma_bar
