# encoding: utf-8
"""
leuvenmapmatching.util.kalman
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging

from pykalman import KalmanFilter
import numpy as np


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def smooth_path(path, dt=1, obs_noise=1e-4, loc_var=1e-4, vel_var=1e-6, kf=None,
                rm_outliers=False, use_euclidean=True, n_iter=1000):
    """Apply Kalman filtering. Assumes data with a constant sample rate.

    Inspired by https://github.com/FlorianWilhelm/gps_data_with_python

    :param path:
    :param dt: Sample interval in seconds
    :param obs_noise: Observation noise (default=1e-4, approx 10-30m)
    :param loc_var: estimated location variance
    :param vel_var: estimated velocity variance
    :param kf: Trained Kalman filter
    :param rm_outliers: Remove outliers based on Kalman prediction
        True or 1 will be removal, 2 will also retrain after removal
    :param use_euclidean:
    :param n_iter: Kalman iterations
    :return:
    """
    path = np.array(path)
    if kf is None:
        # state is (x, y, v_x, v_y)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0,  dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])

        # observations is (x, y)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        R = np.diag([obs_noise, obs_noise]) ** 2

        initial_state_mean = np.hstack([path[0, :2], 2 * [0.]])
        initial_state_covariance = np.diag([loc_var, loc_var, vel_var, vel_var]) ** 2

        kf = KalmanFilter(transition_matrices=F,
                          observation_matrices=H,
                          observation_covariance=R,
                          initial_state_mean=initial_state_mean,
                          initial_state_covariance=initial_state_covariance,
                          em_vars=['transition_covariance'])

        if n_iter > 0:
            logger.debug("Start learning")
            kf = kf.em(path[:, :2], n_iter=n_iter)

    state_means, state_vars = kf.smooth(path[:, :2])

    if use_euclidean:
        from .dist_euclidean import distance
        distance_f = distance
    else:
        from .dist_latlon import distance
        distance_f = distance
    if rm_outliers:
        path_ma = np.ma.asarray(path[:, :2])
        for idx in range(path.shape[0]):
            d = distance_f(path[idx, :2], state_means[idx, :2])
            if d > obs_noise * 2:
                logger.debug("Rm point {}".format(idx))
                path_ma[idx] = np.ma.masked
        if rm_outliers == 2:
            logger.debug("Retrain")
            kf = kf.em(path_ma, n_iter=n_iter)
        state_means, state_vars = kf.smooth(path_ma)

    return state_means, state_vars, kf
