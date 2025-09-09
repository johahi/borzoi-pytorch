from setuptools import setup, find_packages

setup(
    name='borzoi-pytorch',
    version='0.4.4',
    author='Johannes Hingerl',
    author_email='johannes.hingerl@tum.de',
    packages=['borzoi_pytorch'],
    # packages = find_packages(exclude=[]),
    include_package_data = True,
    url='https://github.com/johahi/borzoi-pytorch',
    license='LICENSE',
    description='The Borzoi model from Linder et al., but in Pytorch',
    install_requires=[
        "einops >= 0.5",
        "numpy >= 1.14.2",
        "torch >= 2.1.0",
        "transformers >= 4.34.1,<4.51.0",
        "jupyter >= 1.0.0; extra == 'dev'",
	"intervaltree~=3.1.0",
	"pandas",
        "flash-attn >= 2.6.3; extra == 'flash'"
    ],
)
