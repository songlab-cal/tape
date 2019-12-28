# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

setup(
    name='tape_proteins',
    packages=find_packages(),
    version='0.3',
    description="Repostory of Protein Benchmarking and Modeling",
    author="Roshan Rao, Nick Bhattacharya, Neil Thomas",
    author_email='roshan_rao@berkeley.edu, nickbhat@berkeley.edu, nthomas@berkeley.edu',
    url='https://github.com/songlab-cal/tape',
    license=LICENSE,
    keywords=['Proteins', 'Deep Learning', 'Pytorch', 'TAPE'],
    include_package_data=True,
    install_requires=[
        'torch>=1.0',
        'tqdm',
        'tensorboardX',
        'scipy',
        'lmdb',
        'boto3',
        'requests',
        'biopython',
    ],
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
