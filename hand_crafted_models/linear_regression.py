import numpy as np

from hand_crafted_models.loss_functions import mean_squared_error
from hand_crafted_models.optimization import (
    gradient_descent, closed_form_linear_algebra, GradientStep, WeightsAndBias,
)
from hand_crafted_models.utils import forward_pass


def _step(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        one: np.ndarray
) -> GradientStep:
    # Get predictions
    y_hat = forward_pass(x=x, weights=weights, bias=bias)
    # Calculate total loss value for current parameter values (e.g., MSE cost fn)
    loss = mean_squared_error(y_hat=y_hat, y=y)
    # Perform back-propagation to parameters
    # The MSE derivative
    d_y = 2 * (y - y_hat)  # [B, 1]
    # The Weights derivative w.r.t. MSE loss fn
    d_w = d_y.T @ x  # [1, B] @ [B, N] -> [1, N]
    # The Bias derivative w.r.t. MSE loss fn
    d_b = d_y.T @ one  # [B, 1] -> [1]
    return loss, d_w, d_b


def get_beta_sgd(
        x: np.ndarray,
        y: np.ndarray,
        lr: float = 0.001,
        tol: float = 1e-6,
        max_grad: float = 10.0,
        max_loops: int = 10000
) -> WeightsAndBias:
    return gradient_descent(
        x=x,
        y=y,
        fn=_step,
        lr=lr,
        tol=tol,
        max_grad=max_grad,
        max_loops=max_loops
    )


def get_beta_linalg(
        x: np.ndarray,
        y: np.ndarray,
        add_bias: bool = True
) -> WeightsAndBias:
    return closed_form_linear_algebra(
        x=x,
        y=y,
        add_bias=add_bias
    )
