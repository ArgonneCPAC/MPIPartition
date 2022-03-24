#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = []

test_requirements = []

setup(
    author="Michael Buehlmann",
    author_email="buehlmann.michi@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description=" MPI volume decomposition and particle distribution tools",
    entry_points={},
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="mpipartition",
    name="mpipartition",
    packages=find_packages(include=["mpipartition", "mpipartition.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ArgonneCPAC/mpipartition",
    version="0.2.1",
    zip_safe=False,
)
