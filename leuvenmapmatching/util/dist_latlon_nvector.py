# encoding: utf-8
"""
leuvenmapmatching.util.dist_latlon_nvector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import math

import numpy as np
from nvector._core import unit, n_E2lat_lon, great_circle_normal
import nvector as nv


frame = nv.FrameE(a=6371e3, f=0)
logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def distance(p1, p2):
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


def distance_gp(p1, p2):
    d, _, _ = p1.distance_and_azimuth(p2)
    return d


def distance_point_to_segment(p, s1, s2, delta=0.0):
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
    """Distance between segments. If no intersection within range, simplified to distance from f2 to [t1,t2].

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on t)
    """
    # TODO: Should be improved
    f1_gp = frame.GeoPoint(f1[0], f1[1], degrees=True)
    f2_gp = frame.GeoPoint(f2[0], f2[1], degrees=True)
    path_f = nv.GeoPath(f1_gp, f2_gp)
    t1_gp = frame.GeoPoint(t1[0], t1[1], degrees=True)
    t2_gp = frame.GeoPoint(t2[0], t2[1], degrees=True)
    path_t = nv.GeoPath(t1_gp, t2_gp)
    p_int = path_f.intersect(path_t)
    p_int_gp = p_int.to_geo_point()
    if path_f.on_path(p_int)[0] and path_t.on_path(p_int)[0]:
        # Intersection point is on segments, between both begins and ends
        loc = (p_int_gp.latitude_deg[0], p_int_gp.longitude_deg[0])
        u_f = distance_gp(f1_gp, p_int_gp) / distance_gp(f1_gp, f2_gp)
        u_t = distance_gp(t1_gp, p_int_gp) / distance_gp(t1_gp, t2_gp)
        return 0, loc, loc, u_f, u_t
    # No intersection, use last point of map segment (the assumption is the observations are far apart)
    # TODO: decide which point to use (see distance_segment_to_segment)
    p_int, u_t = _project_nvector(t1_gp, t2_gp, f2_gp)
    u_f = 1
    d, _, _ = f2_gp.distance_and_azimuth(p_int)
    return d, (f1, f2), (p_int_gp.latitude_deg[0], p_int_gp.longitude_deg[0]), u_f, u_t


def project(s1, s2, p, delta=0.0):
    s1 = frame.GeoPoint(s1[0], s1[1], degrees=True)
    s2 = frame.GeoPoint(s2[0], s2[1], degrees=True)
    p = frame.GeoPoint(p[0], p[1], degrees=True)
    p_int, ti = _project_nvector(s1, s2, p, delta=delta)
    return (p_int.latitude_deg[0], p_int.longitude_deg[0]), ti


def _project_nvector(s1, s2, p, delta=0.0):
    path = nv.GeoPath(s1, s2)
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
    return nv.GeoPoint(lat_C, long_C, frame=frame)


def interpolate_path(path, dd):
    """
    TODO: interplate time as third term
    :param path: (lat, lon)
    :param dd: Distance difference (meter)
    :return:
    """
    path_new = [path[0]]
    for p1, p2 in zip(path, path[1:]):
        dist = distance(p1, p2)
        if dist > dd:
            s1 = frame.GeoPoint(p1[0], p1[1], degrees=True)
            s2 = frame.GeoPoint(p2[0], p2[1], degrees=True)
            segment = nv.GeoPath(s1, s2)
            dt = int(math.floor(dist / dd))
            for dti in range(1, dt):
                p_new = segment.interpolate(dti/dt).to_geo_point()
                path_new.append((p_new.latitude_deg[0], p_new.longitude_deg[0]))
        path_new.append(p2)
    return path_new
