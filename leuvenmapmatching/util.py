# encoding: utf-8
"""

Created by Wannes Meert.
Copyright (c) 2015-2017 DTAI, KU Leuven. All rights reserved.
"""
import math
from itertools import islice

import nvector
from nvector._core import unit, n_E2lat_lon, great_circle_normal
import nvector as nv
import numpy as np
import gpxpy
import gpxpy.gpx
import logging
# from filterpy.kalman import KalmanFilter
from pykalman import KalmanFilter


phi_er = 0
lambda_er = 0
frame = nvector.FrameE(a=6371e3, f=0)
logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def latlon2equirectangular(lat, lon):
    """Naive equirectangular projection.  This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.
    """
    x = (lon - lambda_er) * math.cos(phi_er)
    y = lat - phi_er
    return y, x


def equirectangular2latlon(y, x):
    """Naive equirectangular projection. This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.
    """
    lon = x / math.cos(phi_er) + lambda_er
    lat = y + phi_er
    return lat, lon


def distance(p1, p2):
    result = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    # print("distance({}, {}) -> {}".format(p1, p2, result))
    return result


def distance_latlon(p1, p2):
    """

    :param p1:
    :param p2:
    :return: Distance in meters
    """
    p1 = frame.GeoPoint(p1[0], p1[1], degrees=True)
    p2 = frame.GeoPoint(p2[0], p2[1], degrees=True)
    d, _, _ = p1.distance_and_azimuth(p2)
    # print("distance_latlon({}, {}) -> {}".format(p1, p2, d))
    return d


def distance_point_to_segment(p, s1, s2, delta=0.0):
    p_int, ti = project(s1, s2, p, delta=delta)
    return distance(p_int, p), p_int, ti
    # l1a = np.array(s1)
    # l2a = np.array(s2)
    # pa = np.array(p)
    # return np.linalg.norm(np.cross(l2a - l1a, l1a - pa)) / np.linalg.norm(l2a - l1a)


def distance_point_to_segment_latlon(p, s1, s2, delta=0.0):
    """
    TODO: A point exactly on the line gives an error.

    :param s1: Segment start point
    :param s2: Segment end point
    :param p: Point to measure distance from path to
    :param delta: Stay away from the endpoints with this factor
    :return: (Distance in meters, projected location on segmegnt)
    """
    # TODO: Initialize all points as GeoPoint when loading data
    s1 = frame.GeoPoint(s1[0], s1[1], degrees=True)
    s2 = frame.GeoPoint(s2[0], s2[1], degrees=True)
    p = frame.GeoPoint(p[0], p[1], degrees=True)
    p_int, ti = _project_nvector(s1, s2, p)
    d, _, _ = p.distance_and_azimuth(p_int)
    return d, (p_int.latitude_deg[0], p_int.longitude_deg[0]), ti


def distance_segment_to_segment(f1, f2, t1, t2):
    """Distance between segments..

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on t)
    """
    x1, y1 = f1
    x2, y2 = f2
    x3, y3 = t1
    x4, y4 = t2
    n = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if np.allclose([n], [0]):
        # parallel
        is_parallel = True
        n = 0.0001  # TODO: simulates a point far away
    else:
        is_parallel = False
    u_f = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / n
    u_t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / n
    xi = x1 + u_f * (x2 - x1)
    yi = y1 + u_f * (y2 - y1)
    changed_f = False
    changed_t = False
    if u_t > 1:
        u_t = 1
        changed_t = True
    elif u_t < 0:
        u_t = 0
        changed_t = True
    if u_f > 1:
        u_f = 1
        changed_f = True
    elif u_f < 0:
        u_f = 0
        changed_f = True
    if not changed_t and not changed_f:
        return 0, (xi, yi), (xi, yi), u_t, u_f
    xf = x1 + u_f * (x2 - x1)
    yf = y1 + u_f * (y2 - y1)
    xt = x3 + u_t * (x4 - x3)
    yt = y3 + u_t * (y4 - y3)
    if changed_t and changed_f:
        # Compare furthest point from intersection with segment
        df = (xf - xi) ** 2 + (yf - yi) ** 2
        dt = (xt - xi) ** 2 + (yt - yi) ** 2
        if df > dt:
            changed_t = False
        else:
            changed_f = False
    if changed_t:
        pt = (xt, yt)
        pf, u_f = project(f1, f2, pt)
    elif changed_f:
        pf = (xf, yf)
        pt, u_t = project(t1, t2, pf)
    else:
        raise Exception(f"Should not happen")
    d = distance(pf, pt)
    return d, pf, pt, u_f, u_t


def distance_segment_to_segment_latlon(f1, f2, t1, t2):
    """Distance between segments. If no intersection within range, simplified to distance from f2 to [t1,t2].

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on t)
    """
    # TODO: Should be improved
    f1 = frame.GeoPoint(f1[0], f1[1], degrees=True)
    f2 = frame.GeoPoint(f2[0], f2[1], degrees=True)
    path_f = nv.GeoPath(f1, f2)
    t1 = frame.GeoPoint(t1[0], t1[1], degrees=True)
    t2 = frame.GeoPoint(t2[0], t2[1], degrees=True)
    path_t = nv.GeoPath(t1, t2)
    p_int = path_f.intersect(path_t)
    if path_f.on_path(p_int)[0] and path_t.on_path(p_int)[0]:
        # Intersection point is on segments, between both begins and ends
        loc = (p_int.latitude_deg[0], p_int.longitude_deg[0])
        u_f = distance_latlon(f1, loc) / distance_latlon(f1, f2)
        u_t = distance_latlon(t1, loc) / distance_latlon(t1, t2)
        return 0, loc, loc, u_f, u_t
    # No intersection, use last point of map segment (the assumption is the observations are far apart)
    # TODO: decide which point to use (see distance_segment_to_segment)
    p_int, u_t = _project_nvector(t1, t2, f2)
    u_f = 1
    d, _, _ = f2.distance_and_azimuth(p_int)
    return d, (f1, f2), (p_int.latitude_deg[0], p_int.longitude_deg[0]), u_f, u_t


def project(s1, s2, p, delta=0.0):
    if np.isclose(s1[0], s2[0]) and np.isclose(s1[1], s2[1]):
        return s1, distance(s1, p)

    l2 = (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2
    t = max(delta, min(1-delta, ((p[0]-s1[0])*(s2[0]-s1[0]) + (p[1]-s1[1])*(s2[1]-s1[1])) / l2))
    return (s1[0] + t * (s2[0]-s1[0]), s1[1] + t * (s2[1]-s1[1])), t


def project_latlon(s1, s2, p, delta=0.0):
    s1 = frame.GeoPoint(s1[0], s1[1], degrees=True)
    s2 = frame.GeoPoint(s2[0], s2[1], degrees=True)
    p = frame.GeoPoint(p[0], p[1], degrees=True)
    p_int, ti = _project_nvector(s1, s2, p, delta=delta)
    return (p_int.latitude_deg[0], p_int.longitude_deg[0]), ti


def _project_nvector(s1, s2, p, delta=0.0):
    path = nvector.GeoPath(s1, s2)
    p_intr = _cross_track_point(path, p)
    pin = p_intr.to_nvector().normal
    s1n = s1.to_nvector().normal
    s2n = s2.to_nvector().normal
    ti = np.linalg.norm(pin - s1n) / np.linalg.norm(s2n - s1n)
    ti = max(delta, min(1 - delta, ti))
    return path.interpolate(ti).to_geo_point(), ti


def _cross_track_point(path, point):
    """Extend nvector package to find the projection point.

    The projection point is the closest point on path to the given point.
    Based on the nvector.cross_track_distance function.
    http://www.navlab.net/nvector/

    :param path: GeoPath
    :param point: GeoPoint
    """
    c_E = great_circle_normal(*path.nvector_normals())
    n_EB_E = point.to_nvector().normal  # type: np.array
    c_EP_E = np.cross(c_E, n_EB_E, axis=0)

    # Find intersection point C that is closest to point B
    frame = path.positionA.frame
    n_EA1_E = path.positionA.to_nvector().normal  # should also be ok to use  n_EB_C
    n_EC_E_tmp = unit(np.cross(c_E, c_EP_E, axis=0), norm_zero_vector=np.nan)
    n_EC_E = np.sign(np.dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp
    if np.any(np.isnan(n_EC_E)):
        raise Exception('Paths are Equal. Intersection point undefined. NaN returned.')
    lat_C, long_C = n_E2lat_lon(n_EC_E, frame.R_Ee)
    return nvector.GeoPoint(lat_C, long_C, frame=frame)


def gpx_to_path(gpx_file):
    gpx_fh = open(gpx_file)
    track = None
    try:
        gpx = gpxpy.parse(gpx_fh)
        if len(gpx.tracks) == 0:
            logger.error('No tracks found in GPX file (<trk> tag missing?): {}'.format(gpx_file))
            return None
        logger.info("Read gpx file: {} points, {} tracks, {} segments".format(gpx.get_points_no(), len(gpx.tracks),
                                                                               len(gpx.tracks[0].segments)))
        track = [(p.latitude, p.longitude, p.time) for p in gpx.tracks[0].segments[0].points]
    finally:
        gpx_fh.close()
    return track


def path_to_gpx(path, filename=None):
    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    gpx_segment.points = [(gpxpy.gpx.GPXTrackPoint(lat, lon, time=time)) for (lat, lon, time) in path]

    if filename:
        with open(filename, 'w') as gpx_fh:
            gpx_fh.write(gpx.to_xml())

    return gpx

def interpolate_path(path, dd, use_latlon=True):
    """
    TODO: interplate time as third term
    :param path: (lat, lon)
    :param dd: Distance difference (meter)
    :return:
    """
    path_new = [path[0]]
    for p1, p2 in zip(path, path[1:]):
        if use_latlon:
            dist = distance_latlon(p1, p2)
        else:
            dist = distance(p1, p2)
        if dist > dd:
            s1 = frame.GeoPoint(p1[0], p1[1], degrees=True)
            s2 = frame.GeoPoint(p2[0], p2[1], degrees=True)
            segment = nvector.GeoPath(s1, s2)
            dt = int(math.floor(dist / dd))
            for dti in range(1, dt):
                p_new = segment.interpolate(dti/dt).to_geo_point()
                path_new.append((p_new.latitude_deg[0], p_new.longitude_deg[0]))
        path_new.append(p2)
    return path_new


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
        distance_f = distance
    else:
        distance_f = distance_latlon
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


def approx_equal(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))


class Segment(object):
    """Segment and interpolated point"""
    __slots__ = ["l1", "p1", "l2", "p2", "_pi", "_ti"]

    def __init__(self, l1, p1, l2=None, p2=None, pi=None, ti=None):
        self.l1 = l1  # Start of segment
        self.p1 = p1
        self.l2 = l2  # End of segment, if None the segment is a point
        self.p2 = p2
        self.pi = pi  # Interpolated point
        self.ti = ti  # Position on segment p1-p2

    @property
    def label(self):
        return f"{self.l1}-{self.l2}"

    @property
    def key(self):
        return f"{self.l1}-{self.l2}"

    @property
    def pi(self):
        if self.p2 is None:
            return self.p1
        return self._pi

    @pi.setter
    def pi(self, value):
        if value is not None and len(value) > 2:
            self._pi = tuple(value[:2])
        else:
            self._pi = value

    @property
    def ti(self):
        if self.p2 is None:
            return 0
        return self._ti

    @ti.setter
    def ti(self, value):
        self._ti = value

    def is_point(self):
        return self.p2 is None

    def __str__(self):
        if self.p2 is None:
            return f"{self.l1}"
        if self._pi is not None:
            return f"{self.l1}-i-{self.l2}"
        return f"{self.l1}-{self.l2}"

    def __repr__(self):
        return self.__str__()
