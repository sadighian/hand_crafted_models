import os

from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cwd, 'requirements.txt')
with open(file) as f:
    dependencies = list(map(lambda x: x.replace("\n", " "), f.readlines()))

setup(
    name='hand_crafted_models',
    version='1.0',
    description='Hand-crafted discriminative linear predictive models',
    long_description="Implementation of linear regression, logistic regression, "
                     "and gradient descent by hand.",
    author='Jonathan SADIGHIAN',
    url='https://github.com/sadighian',
    install_requires=dependencies,
    packages=find_packages(),
)
