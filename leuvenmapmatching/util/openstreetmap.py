# encoding: utf-8
"""
leuvenmapmatching.util.openstreetmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
from pathlib import Path
import requests
import tempfile
import osmread

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def locations_to_map(locations, map_con, filename=None):
    lats, lons = zip(*locations)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    bb = [lon_min, lat_min, lon_max, lat_max]
    return bb_to_map(bb, map_con, filename)


def bb_to_map(bb, map_con, filename=None):
    """

    :param bb: [lon_min, lat_min, lon_max, lat_max]
    :param map:
    :param filename:
    :return:
    """
    if filename:
        xml_file = Path(filename)
    else:
        xml_file = Path(tempfile.gettempdir()) / "osm.xml"
    if not xml_file.exists():
        bb_str = ",".join(str(coord) for coord in bb)
        url = 'http://overpass-api.de/api/map?bbox='+bb_str
        logger.debug("Downloading {} from {} ...".format(xml_file, url))
        r = requests.get(url, stream=True)
        with xml_file.open('wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)
        logger.debug("... done")
    else:
        logger.debug("Reusing existing file: {}".format(xml_file))
    return file_to_map(xml_file, map_con)


def file_to_map(filename, map_con):
    logger.debug("Parse OSM file ...")
    for entity in osmread.parse_file(str(filename)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, node_b)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    logger.debug("... done")
    logger.debug("Purging database ...")
    map_con.purge()
    logger.debug("... done")
