import numpy as np

# conventions
# _t-1
# t_


# miu : [[x,   y,   th]
#        [m1x, m1y, s1]
#        [m2x, m2y, s2]
#        [m1x, m1y, s1]
#        [m1x, m1y, s1]].T
# sigma :
# f  : motion model
# Jf : motion model derivative
# h  : observation model
# Jh : observation model derivative
# u  : controls = [v, w].T
# z  : observations = real observations from sensors [x, y, signature] why signature???
# R  : motion noise
# Q  : observation noise


def ekf_slam(miu_, sigma_, u_, z_, N_, R_, Q_):
    # prediction
    Fx = np.hstack((np.eye(3), np.zeros((3, 3 * N_))))

    miu_bar = miu_ + Fx.T @ f(miu_[:, 0], u_)

    Gt = np.eye(3) + Fx.T @ Jf(miu_[:, 0], u_) @ Fx
    sigma_bar = Gt @ sigma_ @ Gt.T + R_

    for i in range(len(z_)):
        new_landmark_miu = np.zeros((3, 1))
        r_i = z_[i, 0]
        ph_i = z_[i, 1]
        s_i = z_[i, 2]
        new_landmark_miu[0, 0] = miu_bar[1, 0] + r_i * np.cos(miu_bar[0, 0] + ph_i)
        new_landmark_miu[1, 0] = miu_bar[2, 0] + r_i * np.sin(miu_bar[0, 0] + ph_i)
        new_landmark_miu[2, 0] = s_i

        mah_dist = np.zeros((1, N_ + 1))
        for k in range(N_ + 1):
            z_expected_k = h(miu_bar[:, 0], miu_bar[:, k])

            Fxk = np.zeros(6, 3(N_ + 3))
            Fxk[:3, :3] = np.eye(3)
            Fxk[3:, 3 + 3 * k : 6 + 3 * k] = np.eye(3)

            Hk = H(miu_bar[:, 0], miu_bar[:, k]) @ Fxk

            psi_k = Hk @ sigma_bar @ Hk.T + Q_

            # Mahalanobis Distance
            mah_dist[k] = (z_[i, :].T - z_expected_k).T @ np.linalg.pinv(psi_k) @ (z_[i, :].T - z_expected_k)

        mah_dist[N_ + 1] = mahalanobis_distance_threshold

        j = np.argmin(mah_dist)  # ML correspondence selection

        N_ = np.max(N_, j)

        if N_ == j:
            miu_bar = np.hstack(miu_bar, new_landmark_miu)

            sigma_bar = np.vstack(
                (np.hstack((sigma_bar, np.zeros((sigma_bar.shape[0], 3)))), np.hstack((np.zeros((3, sigma_bar.shape[1])), np.eye(3))))
            )

        Fxj = np.zeros(6, 3(N_ + 3))
        Fxj[:3, :3] = np.eye(3)
        Fxj[3:, 3 + 3 * j : 6 + 3 * j] = np.eye(3)

        Hj = H(miu_bar[:, 0], miu_bar[:, j]) @ Fxj
        psi_j = Hj @ sigma_bar @ Hj.T + Q_

        K = sigma_bar @ Hj.T @ np.linalg.pinv(psi_j)

        z_expected_j = h(miu_bar[:, 0], miu_bar[:, j])

        miu_bar = miu_bar + K @ (z_[i, :] - z_expected_j)
        sigma_bar = (np.eye(np.shape(sigma_bar)) - K @ Hj) @ sigma_bar

    return miu_bar, sigma_bar
