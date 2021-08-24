import numpy as np

from hand_crafted_models.activations import sigmoid
from hand_crafted_models.loss_functions import log_loss
from hand_crafted_models.optimization import gradient_descent, GradientStep, WeightsAndBias
from hand_crafted_models.utils import forward_pass


def _step(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        one: np.ndarray
) -> GradientStep:
    """
    Log-loss: Calculate gradients for a given step.
    
    :param x: Input data [Batch, Features]
    :param y: Label data [Batch, 1]
    :param weights: Feature parameters [1, Features]
    :param bias: Bias parameter [1, 1]
    :param one: (Ignore) Vector of ones
    :return: step loss, weight gradients, bias gradient
    """
    # Make prediction
    y_hat = sigmoid(logits=forward_pass(x=x, weights=weights, bias=bias))
    # Derive total loss value for current parameter values (i.e., Binary Cross-Entropy loss cost fn)
    loss = log_loss(y_hat=y_hat, y=y)
    # Perform back-propagation to parameters
    d_y = (y - y_hat)  # [B, 1]
    # Weights derivative w.r.t. Log-Loss loss fn
    d_w = d_y.T @ x  # [1, B] @ [B, N] -> [1, N]
    # Bias derivative w.r.t. Log-Loss loss fn
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
    """
    Fit parameters using gradient descent.
    
    :param x: Input data [Batch, Features]
    :param y: Label data [Batch, 1]
    :param lr: Learning rate (i.e., optimizer step size)
    :param tol: Tolerance for early-stopping
    :param max_grad: (Optional) Max size of gradient
    :param max_loops: Maximum number of steps to take
    :return: weight gradients, bias gradient
    """
    return gradient_descent(
        x=x,
        y=y,
        fn=_step,
        lr=lr,
        tol=tol,
        max_grad=max_grad,
        max_loops=max_loops
    )
