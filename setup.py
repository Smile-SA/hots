from setuptools import find_packages, setup

import src.rac

PROJECT = 'rac'

try:
    long_description = open('README.adoc', 'rt').read()
except IOError:
    long_description = ''

setup(
    name=PROJECT,
    version=src.rac.version,

    description='Resource Allocation via Clustering',
    long_description=long_description,

    author_email='rnd@alterway.fr',

    install_requires=[
        'docplex',
        'docplex',
        'matplotlib',
        'numpy',
        'pandas',
        'pathlib',  # to change ?
        'scipy',
        'sklearn',
        'tqdm',
        # for linting
        # 'pep8-naming',
        # 'click',
        # 'click-log',
        # 'pytest',
    ],

    namespace_packages=['src'],
    packages=find_packages(),
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'rac = src.rac.main:main'
        ],
    },

     setup_requires=['numpy'],
    # tests_require=['pytest'],

    # zip_safe=False,
)
