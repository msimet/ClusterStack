from setuptools import setup, find_packages
from io import open

# read the contents of the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='clusterstack',
    version='0.1.0',
    description='Build models for stacked galaxy cluster lensing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/msimet/clusterstack',
    author='M. Simet',
    packages=['clusterstack'],
    install_requires=['numpy'],
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
