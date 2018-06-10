from setuptools import setup

setup(
    name='sc2scout',
    version='0.0.1',
    description='SC2SCOUT',
    keywords='SC2SCOUT',
    packages=[
        'sc2scout',
        'sc2scout.envs',
        'sc2scout.agents',
        'sc2scout.wrapper',
    ],

    install_requires=[
        'pysc2',
        'gym',
        'numpy',
    ],
)
