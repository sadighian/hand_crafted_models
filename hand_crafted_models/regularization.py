from typing import Callable, Tuple

import numpy as np

NormFn = Callable[[np.ndarray, float], float]
NormDLoss = Callable[[np.ndarray, float], float]


def l1(
        weights: np.ndarray,
        scale: float
) -> float:
    """
    L1-Norm loss.

    :param weights: Parameters (i.e., beta coefficients [Batch, Features])
    :param scale: Value used to scale the loss (also known as 'lambda')
    :return: L1 loss
    """
    total = 0.0
    weights = weights.flatten()  # [1, Features] -> [Features]
    for w in weights:
        total += abs(w)
    loss = scale * total
    return loss


def l2(
        weights: np.ndarray,
        scale: float
) -> float:
    """
    L2-Norm loss.

    :param weights: Parameters (i.e., beta coefficients [Batch, Features])
    :param scale: Value used to scale the loss (also known as 'lambda')
    :return: L2 loss
    """
    total = 0.0
    weights = weights.flatten()
    for w in weights:
        total += w ** 2
    loss = scale * total * 0.5
    return loss


def l1_gradient(
        weights: np.ndarray,
        scale: float
) -> float:
    """
    L1-Norm loss for back-propagation.

    :param weights: Parameters (i.e., beta coefficients [Batch, Features])
    :param scale: Value used to scale the loss (also known as 'lambda')
    :return: L1-loss to be back-propagated
    """
    gradient = 0.0
    weights = weights.flatten()
    for w in weights:
        if w > 0.0:
            sign = 1.0
        elif w == 0.0:
            sign = 0.0
        else:
            sign = -1.0
        gradient += sign
    loss = scale * gradient
    return loss


def l2_gradient(
        weights: np.ndarray,
        scale: float
) -> float:
    """
    L2-Norm loss for back-propagation.

    :param weights: Parameters (i.e., beta coefficients [Batch, Features])
    :param scale: Value used to scale the loss (also known as 'lambda')
    :return: L2-loss to be back-propagated
    """
    gradient = 0.0
    weights = weights.flatten()
    for w in weights:
        gradient += w
    # loss = 0.50 * scale * gradient
    loss = scale * gradient
    return loss


def get_regularization_fn(
        name: str
) -> Tuple[NormFn, NormDLoss]:
    """
    Import regularization function and its respective derivative loss function.
    
    :param name: l1 or l2 norm
    :return: reg function, reg. derivative fn
    """
    name = name.lower()
    if name == 'l1':
        return l1, l1_gradient
    elif name == 'l2':
        return l2, l2_gradient
    raise ValueError(f'Unknown "name" provided ("{name}")')
