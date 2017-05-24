#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='bb_utils',
    version='0.1',
    description='Beesbook utils',
    author='Leon Sixt',
    author_email='mail@leon-sixt.de',
    url='https://github.com/BioroboticsLab/bb_utils/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['bb_utils'],
    package_dir={'bb_utils': 'bb_utils/'},
    package_data={'bb_utils': ['data/hatchdates2016.csv']},
    entry_points={
        'console_scripts': [
            'bb_gt_to_hdf5 = bb_utils.scripts.gt_to_hdf5:run',
            'shuffle_hdf5 = bb_utils.scripts.shuffle_hdf5:main',
        ]
    },
    scripts=[
        'scripts/shuffle_all_hdf5.sh',
        'scripts/build_gt.sh',
    ]
)
