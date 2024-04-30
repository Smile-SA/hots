"""Package for testing containers resource allocation via clustering."""
import logging
from importlib import metadata


# Custom logger
LOG = logging.getLogger(name=__name__)

# PEP 396 style version marker
try:
    __version__ = metadata.version('hots')
except metadata.PackageNotFoundError:
    print('Package `hots` not found.')
    LOG.warning('Could not get the package version from metadata')
    __version__ = 'unknown'
except Exception as e:
    print(f'An error occurred: {e}')

__author__ = 'Smile R&D team'
__author_email__ = 'rnd@smile.fr'
__license__ = 'MIT'
