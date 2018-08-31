# encoding: utf-8
"""
leuvenmapmatching
~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
from . import map, matcher, util
# visualization is not loaded by default (avoid loading unnecessary dependencies such as matplotlib).

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")

__version__ = '0.5.1'
