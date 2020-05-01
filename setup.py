# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os


def get_version():
    directory = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(directory, 'tape', '__init__.py')
    with open(init_file) as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

with open('requirements.txt', 'r') as reqs:
    requirements = reqs.read().split()

setup(
    name='tape_proteins',
    packages=find_packages(),
    version=get_version(),
    description="Repostory of Protein Benchmarking and Modeling",
    author="Roshan Rao, Nick Bhattacharya, Neil Thomas",
    author_email='roshan_rao@berkeley.edu, nickbhat@berkeley.edu, nthomas@berkeley.edu',
    url='https://github.com/songlab-cal/tape',
    license=LICENSE,
    keywords=['Proteins', 'Deep Learning', 'Pytorch', 'TAPE'],
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'tape-train = tape.main:run_train',
            'tape-train-distributed = tape.main:run_train_distributed',
            'tape-eval = tape.main:run_eval',
            'tape-embed = tape.main:run_embed',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
