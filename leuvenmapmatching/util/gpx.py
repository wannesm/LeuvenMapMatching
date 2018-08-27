# encoding: utf-8
"""
leuvenmapmatching.util.gpx
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some additional functions to interact with the gpx library.

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging

import gpxpy
import gpxpy.gpx


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def gpx_to_path(gpx_file):
    gpx_fh = open(gpx_file)
    track = None
    try:
        gpx = gpxpy.parse(gpx_fh)
        if len(gpx.tracks) == 0:
            logger.error('No tracks found in GPX file (<trk> tag missing?): {}'.format(gpx_file))
            return None
        logger.info("Read gpx file: {} points, {} tracks, {} segments".format(
            gpx.get_points_no(), len(gpx.tracks), len(gpx.tracks[0].segments)))
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
