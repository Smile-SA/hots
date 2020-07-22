"""
===
cots
===

Package for testing resource allocation via clustering.
"""
import logging

import pkg_resources

# Custom logger
LOG = logging.getLogger(name=__name__)

# PEP 396 style version marker
try:
    __version__ = pkg_resources.get_distribution('cots').version
except pkg_resources.DistributionNotFound:
    LOG.warning('Could not get the package version from pkg_resources')
    __version__ = 'unknown'

__author__ = 'AlterWay R&D team'
__author_email__ = 'rnd@alterway.fr'
__license__ = 'MIT'
