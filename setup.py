from setuptools import setup

setup(
    name='borzoi-pytorch',
    version='0.0.1',
    author='',
    author_email='',
    packages=['borzoi_pytorch'],
    url='https://github.com/johahi/borzoi-pytorch',
    license='LICENSE',
    description='The Borzoi model from Linder et al., but in Pytorch',
    install_requires=[
        "einops >= 0.5",
        "numpy >= 1.14.2",
        "torch >= 1.13.0",
    ],
)
