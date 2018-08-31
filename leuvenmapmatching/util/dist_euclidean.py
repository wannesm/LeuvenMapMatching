# encoding: utf-8
"""
leuvenmapmatching.util.dist_euclidean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import math

import numpy as np

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def distance(p1, p2):
    result = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    # print("distance({}, {}) -> {}".format(p1, p2, result))
    return result


def distance_point_to_segment(p, s1, s2, delta=0.0):
    p_int, ti = project(s1, s2, p, delta=delta)
    return distance(p_int, p), p_int, ti
    # l1a = np.array(s1)
    # l2a = np.array(s2)
    # pa = np.array(p)
    # return np.linalg.norm(np.cross(l2a - l1a, l1a - pa)) / np.linalg.norm(l2a - l1a)


def distance_segment_to_segment(f1, f2, t1, t2):
    """Distance between segments..

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on f, rel pos on t)
    """
    x1, y1 = f1
    x2, y2 = f2
    x3, y3 = t1
    x4, y4 = t2
    n = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if np.allclose([n], [0], rtol=0):
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


def project(s1, s2, p, delta=0.0):
    """

    :param s1: Segment start
    :param s2: Segment end
    :param p: Point
    :param delta: Keep delta fraction away from ends
    :return: Point of projection, Relative position on segment
    """
    if np.isclose(s1[0], s2[0], rtol=0) and np.isclose(s1[1], s2[1], rtol=0):
        return s1, 0.0

    l2 = (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2
    t = max(delta, min(1-delta, ((p[0]-s1[0])*(s2[0]-s1[0]) + (p[1]-s1[1])*(s2[1]-s1[1])) / l2))
    return (s1[0] + t * (s2[0]-s1[0]), s1[1] + t * (s2[1]-s1[1])), t


def interpolate_path(path, dd):
    """
    TODO: interplate time as third term
    :param path: (y, x)
    :param dd: Distance difference (meter)
    :return:
    """
    path_new = [path[0]]
    for p1, p2 in zip(path, path[1:]):
        dist = distance(p1, p2)
        if dist > dd:
            dt = int(math.ceil(dist / dd))
            dx = (p2[0] - p1[0]) / dt
            dy = (p2[1] - p1[1]) / dt
            px, py = p1[0], p1[1]
            for _ in range(dt):
                px += dx
                py += dy
                path_new.append((px, py))
        path_new.append(p2)
    return path_new


def box_around_point(p, dist):
    lat, lon = p
    lat_t, lon_r = lat + dist, lon + dist
    lat_b, lon_l = lat - dist, lon - dist
    return lat_b, lon_l, lat_t, lon_r


def lines_parallel(la, lb, lc, ld, d=None):
    x1 = la[0] - lb[0]
    y1 = la[1] - lb[1]
    if x1 == 0:
        if y1 == 0:
            return False
        s1 = 0
    else:
        s1 = math.atan(abs(y1 / x1))
    x2 = lc[0] - ld[0]
    y2 = lc[1] - ld[1]
    if x2 == 0:
        s2 = 0
        if y2 == 0:
            return False
    else:
        s2 = math.atan(abs(y2 / x2))
    thr = math.pi / 180
    if abs(s1 - s2) > thr:
        return False
    if d is not None:
        dist, _, _, _, _ = distance_segment_to_segment(la, lb, lc, ld)
        if dist > d:
            return False
    return True

