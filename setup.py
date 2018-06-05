from setuptools import setup

setup(
    name='sc2scout',
    version='0.0.1',
    description='SC2SCOUT',
    keywords='SC2SCOUT',
    packages=[
        'sc2scout',
        'sc2scout.envs',
    ],

    install_requires=[
        'pysc2',
        'gym',
        'numpy',
    ],
)
