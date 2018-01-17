#!/usr/bin/env python

from logging import basicConfig, getLogger, INFO

import subprocess
import os

from setuptools import setup, find_packages


basicConfig(level=INFO)
logger = getLogger(__name__)


def validate_requirements():
    res = subprocess.call(args=['which', 'sbt'])
    return True if res == 0 else False


def assemble_scwc():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir('{}/scwc/scwc_base'.format(dir_path))

    assemble_cmd = ['sbt', 'assembly']
    subprocess.run(args=assemble_cmd)

    os.chdir('../../')


if __name__ == '__main__':
    if not validate_requirements():
        raise Exception('Please install sbt first')

    assemble_scwc()

    setup(name='scwc',
          version='0.2.4',
          packages=find_packages(),
          include_package_data=True,
          install_requires=[
              'numpy',
              'scipy',
              'pandas',
              'scikit-learn'
          ])
