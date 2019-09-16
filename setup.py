# -*- coding: utf-8 -*-
from setuptools import setup


with open('README.md', 'r') as rf:
    README = rf.read()

# with open('LICENSE', 'r') as lf:
    # LICENSE = lf.read()

setup(
    name='tape_pytorch',
    version='0.1',
    description='Protein Benchmarking Repository',
    long_description=README,
    author='Roshan Rao, Nick Bhattacharya, Neil Thomas',
    author_email='roshan_rao@berkeley.edu, nickbhat@berkeley.edu, nthomas@berkeley.edu',
    url='https://github.com/rmrao/tape-pytorch',
    # license=LICENSE,
    install_requires=[],
    entry_points={
        'console_scripts': [
            'tape-train = tape_pytorch.main:run_train',
            'tape-eval = tape_pytorch.main:run_eval'
        ]
    },
)
