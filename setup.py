from setuptools import setup, find_packages

setup(
    name='prepare_phase_2',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
