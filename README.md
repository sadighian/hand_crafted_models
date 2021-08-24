# Hand Crafted Models
Simple linear model implementations from scratch.

**Table of contents**
1. Overview
2. Project Structure
3. Getting started
4. Citing this project


## 1. Overview

This repository is intended for educational purposes, where 
popular predictive linear models (i.e., linear and logistic regressions) 
are coded from scratch to demonstrate the mathematics behind the models.


## 2. Project Structure
```
hand_crafted_models/
    activations.py          ...Activation function implementations
    linear_regression.py    ...Linear regression implementations
    logistic_regression.py  ...Logistic regression implementation
    loss_functions.py       ...Loss function implementations
    optimization.py         ...Optimizer implementations
    utils.py                ...Commonly used functions
requirements.txt            ...project dependencies
setup.py            
```

## 3. Getting Started

1.  Clone the project 
```
git clone https://github.com/sadighian/hand_crafted_models.git
```
2.  Create a virtual environment 
```
cd hand_crafted_models          # Change to project directory
virtualenv -p python3 venv      # Create the virtual environment
source venv/bin/activate        # Start using the venv
```

3.  Install the project and its dependencies 
```
pip3 install -e .               # Execute command inside install directory
```

4.  Fit linear models with sample data
```
python3 tests/tests.py
```

## 4. Citing the Project
```
@misc{Hand Crafted Models,
    author = {Jonathan Sadighian},
    title = {Hand Crafted Models: Simple linear model implementations},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/sadighian/hand_crafted_models}},
}
```