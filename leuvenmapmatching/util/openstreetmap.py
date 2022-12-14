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
    """Download map from overpass-api.de.

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


def download_map_xml(fn, bbox, force=False, verbose=False):
    """Download map from overpass-api.de based on a given bbox

    :param fn: Filename where to store the map as xml
    :param bbox: String or array with [lon_min, lat_min, lon_max, lat_max]
    :param force: Also download if file already exists
    :param verbose: Verbose output
    :return:
    """
    fn = Path(fn)
    if type(bbox) is list:
        bb_str = ",".join(str(coord) for coord in bbox)
    elif type(bbox) is str:
        bb_str = bbox
    else:
        raise AttributeError('Unknown type for bbox: {}'.format(type(bbox)))
    if force or not fn.exists():
        if verbose:
            print("Downloading {}".format(fn))
        import requests
        url = f'http://overpass-api.de/api/map?bbox={bb_str}'
        r = requests.get(url, stream=True)
        with fn.open('wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)
    else:
        if verbose:
            print("File already exists")


def create_map_from_xml(fn, include_footways=False, include_parking=False,
                        use_rtree=False, index_edges=False):
    """Create an InMemMap from an OpenStreetMap XML file.

    Used for testing routes on OpenStreetMap.
    """
    from ..map.inmem import InMemMap
    map_con = InMemMap("map", use_latlon=True, use_rtree=use_rtree, index_edges=index_edges)
    cnt = 0
    ways_filter = ['bridleway', 'bus_guideway', 'track']
    if not include_footways:
        ways_filter += ['footway', 'cycleway', 'path']
    parking_filter = ['driveway']
    if not include_parking:
        parking_filter += ['parking_aisle']
    for entity in osmread.parse_file(str(fn)):
        if isinstance(entity, osmread.Way):
            tags = entity.tags
            if 'highway' in tags \
                and not (tags['highway'] in ways_filter) \
                and not ('access' in tags and tags['access'] == 'private') \
                and not ('landuse' in tags and tags['landuse'] == 'square') \
                and not ('amenity' in tags and tags['amenity'] == 'parking') \
                and not ('service' in tags and tags['service'] in parking_filter) \
                and not ('area' in tags and tags['area'] == 'yes'):
                for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                    map_con.add_edge(node_a, node_b)
                    # Some roads are one-way. We'll add both directions.
                    map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()
    return map_con
