import unittest

import numpy as np

from hand_crafted_models.loss_function import log_loss, mean_squared_error
from hand_crafted_models.metric import accuracy
from hand_crafted_models.model.linear_regression import get_beta_linalg, get_beta_sgd
from hand_crafted_models.model.logistic_regression import get_beta_sgd, sigmoid
from hand_crafted_models.utils import (
    gen_regression_data, plot_regression_results, forward_pass, gen_classification_data,
    plot_classification_results,
)


class TestLinearModels(unittest.TestCase):
    
    def test_ols_gradient_descent(self):
        x, y = gen_regression_data(n_features=3)
        weights, bias = get_beta_sgd(x=x, y=y)
        predictions = forward_pass(x, weights, bias)
        loss = mean_squared_error(y_hat=predictions, y=y)
        plot_regression_results(y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        print(f'loss: {loss:,.4f}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_ols_gradient_descent_with_l1_norm(self):
        x, y = gen_regression_data(n_features=10)
        weights, bias = get_beta_sgd(x=x, y=y, regularization='l1', reg_lambda=1e3)
        predictions = forward_pass(x, weights, bias)
        loss = mean_squared_error(y_hat=predictions, y=y)
        # plot_regression_results(y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        print(f'loss: {loss:,.4f}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_ols_linear_algebra(self):
        x, y = gen_regression_data(n_features=1)
        weights, bias = get_beta_linalg(x=x, y=y)
        predictions = forward_pass(x, weights, bias)
        loss = mean_squared_error(y_hat=predictions, y=y)
        plot_regression_results(y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        print(f'loss: {loss:,.4f}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_logistic_regression_gradient_descent(self):
        x, y = gen_classification_data(n_features=2, seed=12345)
        weights, bias = get_beta_sgd(x=x, y=y)
        predictions = sigmoid(logits=forward_pass(x=x, weights=weights, bias=bias))
        loss = log_loss(y_hat=predictions, y=y)
        plot_classification_results(x=x, y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        print(f'loss: {loss:,.4f}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_logistic_regression_gradient_descent_with_norm(self):
        x, y = gen_classification_data(n_features=2, seed=145)
        regularization = 'l2'
        weights, bias = get_beta_sgd(x=x, y=y, regularization=regularization,
                                     reg_lambda=1e-3)
        predictions = sigmoid(logits=forward_pass(x=x, weights=weights, bias=bias))
        loss = log_loss(y_hat=predictions, y=y)
        acc = accuracy(y_hat=predictions, y=y)
        # plot_classification_results(x=x, y=y, y_hat=predictions)
        print(f'accuracy: {acc * 100:,.2f}%')
        print(f'weights: {np.round(weights.flatten(), 2)} | '
              f'sum={np.abs(weights).sum():,.3f}')
        print(f'bias: {bias}')
        print(f'loss: {loss:,.4f}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
