import rac

from setuptools import setup

PROJECT = 'rac'

try:
    long_description = open('README.adoc', 'rt').read()
except IOError:
    long_description = ''

setup(
    name=PROJECT,
    version=rac.version,

    description='Resource Allocation via Clustering',
    long_description=long_description,

    author_email='rnd@alterway.fr',

    install_requires=[
        'docplex',
        'docplex',
        'matplotlib',
        'numpy',
        'pandas',
        'pathlib', # to change ?
        'scipy',
        'sklearn',
        'tqdm',
        # for linting
        # 'flake8-comprehensions',
        # 'flake8-docstrings',
        # 'flake8-import-order',
        # 'flake8-quotes',
        # 'flake8',
        # 'pep8-naming',
        # 'click',
        # 'click-log',
        # 'pytest',
    ],

    packages=['rac'],
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'rac = rac.main:main'
        ],
    },

    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],

    # zip_safe=False,
)
