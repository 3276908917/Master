from setuptools import setup

setup(
    name='cassL',
    version='0.4.4',
    packages=['cassL'],
    package_data={'': [
        'cosmologies.dat'
    ]},
)

# numpy should be <= 1.23.5, otherwise GPy cannot be imported...
