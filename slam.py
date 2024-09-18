import numpy as np
import helpers as hp

MH_DIST_TH = 1000  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,th]
LM_SIZE = 2  # LM state size [x,y]


# motion model
# x = [x,y,th].T
# u = [dx,dy,dth].T
def f(x, u, dt=0.1):
    # v = u[0, 0]
    # w = u[2, 0]

    # return np.array(
    #     [
    #         [(v / w) * (-np.sin(x[2, 0]) + np.sin(x[2, 0] + w * dt))],
    #         [(v / w) * (np.cos(x[2, 0]) - np.cos(x[2, 0] + w * dt))],
    #         [w * dt],
    #     ]
    # )
    return u


# motion model derivative
# x = [x,y,th].T
# u = [dx,dy,dth].T
def Jf(x, u, dt=0.1):
    # v = u[0, 0]
    # w = u[2, 0]

    # arr = np.array(
    #     [
    #         [(v / w) * (-np.cos(x[2, 0]) + np.cos(x[2, 0] + w * dt))],
    #         [(v / w) * (-np.sin(x[2, 0]) + np.sin(x[2, 0] + w * dt))],
    #         [0],
    #     ]
    # )

    # return np.hstack((np.zeros((3, 2)), arr))

    return np.eye(u.shape[0])


# observation model
# x = [x,y,th].T
# lm = [lmx,lmy].T
def h(x, lm):
    d = lm - x[:2, 0].reshape(-1, 1)
    q = (d.T @ d)[0, 0]
    return np.array([[np.sqrt(q)], [hp.angle_dist(np.arctan2(d[1, 0], d[0, 0]), x[2, 0])]])


# observation model derivative
# x = [x,y,th].T
# lm = [lmx,lmy].T
def Jh(x, lm):
    d = lm - x[:2, 0].reshape(-1, 1)
    q = (d.T @ d)[0, 0]

    jac_x = np.zeros((2, 3))
    jac_x[0, 0] = -d[0, 0] / np.sqrt(q)
    jac_x[0, 1] = -d[1, 0] / np.sqrt(q)
    jac_x[1, 0] = d[1, 0] / q
    jac_x[1, 1] = -d[0, 0] / q
    jac_x[1, 2] = -1

    jac_lm = np.zeros((2, 2))
    jac_lm[0, 0] = d[0, 0] / np.sqrt(q)
    jac_lm[0, 1] = d[1, 0] / np.sqrt(q)
    jac_lm[1, 0] = -d[1, 0] / q
    jac_lm[1, 1] = d[0, 0] / q

    return np.hstack((jac_x, jac_lm))


def Fxi(i, N):

    Fx = np.zeros((STATE_SIZE + LM_SIZE, LM_SIZE * N + STATE_SIZE))
    Fx[:STATE_SIZE, :STATE_SIZE] = np.eye(STATE_SIZE)
    if N > 0:
        Fx[
            STATE_SIZE:,
            STATE_SIZE + i * LM_SIZE : STATE_SIZE + (i + 1) * LM_SIZE,
        ] = np.eye(LM_SIZE)
    print(N)
    print(Fx)
    return Fx


# miu : [x, y, th, lm1x, lm1y, lm2x, lm2y, ... ].T
# sigma : covariance matrix


# u  : controls = [v, w].T
# z  : observations = real observations from sensors [lmx, lmy]
# R  : motion noise
# Q  : observation noise
def ekf_slam_step(miu_, sigma_, u_, z_, R_, Q_):

    N_ = int((miu_.shape[0] - STATE_SIZE) / LM_SIZE)
    # prediction
    Fx = np.hstack((np.eye(3), np.zeros((3, LM_SIZE * N_))))

    miu_bar = miu_ + Fx.T @ f(miu_[:, 0].reshape(-1, 1), u_)

    Gt = np.eye(STATE_SIZE + N_ * LM_SIZE) + Fx.T @ Jf(miu_[:, 0].reshape(-1, 1), u_) @ Fx

    sigma_bar = Gt @ sigma_ @ Gt.T + Fx.T @ R_ @ Fx

    # correction
    for i in range(len(z_)):

        new_landmark_miu = np.zeros((2, 1))
        new_landmark_miu[0, 0] = miu_bar[0, 0] + z_[i][0, 0] * np.cos(miu_bar[2, 0] + z_[i][0, 1])
        new_landmark_miu[1, 0] = miu_bar[1, 0] + z_[i][0, 0] * np.sin(miu_bar[2, 0] + z_[i][0, 1])

        mh_dist = []
        for k in range(N_):
            z_expected_k = h(
                miu_bar[:STATE_SIZE, 0].reshape(-1, 1),
                miu_bar[STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE, 0].reshape(
                    -1, 1
                ),
            )

            Fxk = Fxi(k, N_)

            Hk = (
                Jh(
                    miu_bar[:STATE_SIZE, 0].reshape(-1, 1),
                    miu_bar[
                        STATE_SIZE + k * LM_SIZE : STATE_SIZE + (k + 1) * LM_SIZE,
                        0,
                    ].reshape(-1, 1),
                )
                @ Fxk
            )

            psi_k = Hk @ sigma_bar @ Hk.T + Q_

            # Mahalanobis Distance
            mh_dist.append(
                ((z_[i].T - z_expected_k).T @ np.linalg.pinv(psi_k) @ (z_[i].T - z_expected_k))[
                    0, 0
                ]
            )

        mh_dist.append(MH_DIST_TH)

        j = np.argmin(mh_dist)  # ML correspondence selection

        N_ = max(N_, j)
        # create new landmark
        if N_ == j:
            print("New")
            # N_ += 1
            miu_bar = np.vstack((miu_bar, new_landmark_miu))

            sigma_bar = np.vstack(
                (
                    np.hstack((sigma_bar, np.zeros((sigma_bar.shape[0], LM_SIZE)))),
                    np.hstack(
                        (
                            np.zeros((LM_SIZE, sigma_bar.shape[1])),
                            np.eye(LM_SIZE),
                        )
                    ),
                )
            )

        Fxj = Fxi(j, N_)

        Hj = (
            Jh(
                miu_bar[:STATE_SIZE, 0].reshape(-1, 1),
                miu_bar[STATE_SIZE + j * LM_SIZE : STATE_SIZE + (j + 1) * LM_SIZE, 0].reshape(
                    -1, 1
                ),
            )
            @ Fxj
        )

        psi_j = Hj @ sigma_bar @ Hj.T + Q_

        K = sigma_bar @ Hj.T @ np.linalg.pinv(psi_j)

        z_expected_j = h(
            miu_bar[:STATE_SIZE, 0].reshape(-1, 1),
            miu_bar[STATE_SIZE + j * LM_SIZE : STATE_SIZE + (j + 1) * LM_SIZE, 0].reshape(-1, 1),
        )

        miu_bar = miu_bar + K @ (z_[i].T - z_expected_j)
        sigma_bar = (np.eye(np.shape(miu_bar)[0]) - K @ Hj) @ sigma_bar

    return miu_bar, sigma_bar
