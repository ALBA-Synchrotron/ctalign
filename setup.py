#!/usr/bin/env python

from setuptools import setup, find_packages

# ctalign setup.py for usage of setuptools

# The version is updated automatically with bumpversion
# Do not update manually
__version = '2.2.2' 


long_description = """ ctalign project has born with the initial idea of make an 
automatic alignment of tomography projections in order to allow further 
reconstruction.
Nowadays it is used for not only for tomography projection alignment, but also 
for spectroscopic images alignment at ALBA BL09-Mistral beamline.
"""

classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
]


setup(
    name='ctalign',
    version=__version,
    packages=find_packages(),
    entry_points={
	'console_scripts': [
        'ctalign = alignlib.ctalign:main']
    },
    author='Marc Rosanes',
    author_email='mrosanesn@cells.es',
    maintainer='ctgensoft',
    maintainer_email='ctgensoft@cells.es',
    url='https://git.cells.es/controls/ctalign',
    keywords='APP',
    license='GPLv3',
    description='Alignment of images of an hdf5 image stack',
    long_description=long_description,
    requires=['setuptools (>=1.1)'],
    install_requires=['numpy', 'cv2', 'h5py', 'nxs'],
    classifiers=classifiers
    
)


