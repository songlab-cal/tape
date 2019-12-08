# -*- coding: utf-8 -*-
from setuptools import setup


with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

setup(
    name='tape',
    version='0.1',
    description='Repostory of Protein Benchmarking & Modeling',
    long_description=README,
    author='Roshan Rao, Nick Bhattacharya, Neil Thomas',
    author_email='roshan_rao@berkeley.edu, nickbhat@berkeley.edu, nthomas@berkeley.edu',
    url='https://github.com/rmrao/tape-pytorch',
    license=LICENSE,
    keywords=['Proteins', 'Deep Learning', 'Pytorch'],
    install_requires=[
        'torch>=1.0',
        'tqdm',
        'tensorboardX',
        'scipy',
        'lmdb',
    ],
    entry_points={
        'console_scripts': [
            'tape-train = tape.main:run_train',
            'tape-train-distributed = tape.main:run_train_distributed',
            'tape-eval = tape.main:run_eval',
            'tape-embed = tape.main:run_embed',
        ]
    },
)
