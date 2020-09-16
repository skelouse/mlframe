#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'missingno>=0.4.2',
    'pandas>=1.1.1',
    'numpy>=1.19.1',
    'matplotlib>=3.3.1',
    'seaborn>=0.10.1',
    'statsmodels',
    'scipy>=1.3.1',
    'scikit-learn'
]

setup_requirements = []



test_requirements = [ ]

setup(
    author="Sam Stoltenberg",
    author_email='sam@skelouse.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="mlframe package.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='mlframe',
    name='mlframe',
    packages=find_packages(include=['mlframe']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    project_urls={
        'Homepage': "https://github.com/skelouse/mlframe",
        'Download': "https://pypi.org/project/mlframe/",
        'Source Code': "https://github.com/skelouse/mlframe/tree/master/mlframe",
        "Documentation": "https://mlframe.readthedocs.io/en/latest/",
        "Bug Tracker": "https://github.com/skelouse/mlframe/issues",
        "Frame Documentation": "https://mlframe.readthedocs.io/en/latest/api/mlframe.MLFrame.html"
    },
    url='https://github.com/skelouse/mlframe',
    version='0.1.11',
    zip_safe=False,
)
