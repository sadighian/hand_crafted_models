import unittest

from hand_crafted_models.utils import (
    gen_regression_data, plot_regression_results, forward_pass, gen_classification_data,
    plot_classification_results,
)


class TestLinearModels(unittest.TestCase):
    
    def test_ols_gradient_descent(self):
        from hand_crafted_models.linear_regression import get_beta_sgd
        from hand_crafted_models.loss_functions import mean_squared_error
        
        x, y = gen_regression_data(n_features=3)
        weights, bias = get_beta_sgd(x=x, y=y)
        predictions = forward_pass(x, weights, bias)
        loss = mean_squared_error(y_hat=predictions, y=y)
        plot_regression_results(y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_ols_linear_algebra(self):
        from hand_crafted_models.linear_regression import get_beta_linalg
        from hand_crafted_models.loss_functions import mean_squared_error
        
        x, y = gen_regression_data(n_features=1)
        weights, bias = get_beta_linalg(x=x, y=y)
        predictions = forward_pass(x, weights, bias)
        loss = mean_squared_error(y_hat=predictions, y=y)
        plot_regression_results(y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
    
    def test_logistic_regression_gradient_descent(self):
        from hand_crafted_models.logistic_regression import get_beta_sgd, sigmoid
        from hand_crafted_models.loss_functions import log_loss
        
        x, y = gen_classification_data(n_features=2)
        weights, bias = get_beta_sgd(x=x, y=y)
        predictions = sigmoid(logits=forward_pass(x=x, weights=weights, bias=bias))
        loss = log_loss(y_hat=predictions, y=y)
        plot_classification_results(x=x, y=y, y_hat=predictions)
        print(f'weights: {weights}')
        print(f'bias: {bias}')
        self.assertGreater(a=loss, b=0, msg=f'{loss} must be greater than {0}')
