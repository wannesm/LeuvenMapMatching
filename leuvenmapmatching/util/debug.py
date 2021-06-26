import logging


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def printd(*args, **kwargs):
    """Print to debug output."""
    logger.debug(*args, **kwargs)
