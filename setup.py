from setuptools import setup, find_packages

setup(
    name='medddpm',
    version='0.0.1',
    description='install package for 3D paper med-ddpm',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)


