from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression, make_classification


def ensure_dims(
        x: np.ndarray
) -> np.ndarray:
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    return x


def forward_pass(
        x: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray
) -> np.ndarray:
    assert x.ndim == weights.ndim == bias.ndim and x.ndim > 1, \
        f'All dims must match!!! x: {x.ndim}, weights: {weights.ndim}, bias: {bias.ndim}'
    assert x.shape[1] == weights.shape[1], \
        f'Input data (x) and weights must have the same dims'
    return x @ weights.T + bias


def convert_probability_to_boolean(
        y_hat: np.ndarray
) -> np.ndarray:
    predictions = np.round(y_hat).astype(np.int)  # Converts to either: '0' or '1'
    return predictions


def plot_regression_results(
        x: np.ndarray,
        y: np.ndarray,
        y_hat: np.ndarray
) -> None:
    y_hat = ensure_dims(y_hat)
    error = np.mean((y - y_hat) ** 2)
    plt.figure(figsize=(12, 6))
    plt.title(f'Regression Error (i.e., MSE): {error:,.3f}')
    plt.plot(y, label='y', alpha=0.5)
    plt.plot(y_hat, label='y_hat', alpha=0.85, linestyle=':')
    plt.legend()
    plt.show()


def plot_classification_results(
        x: np.ndarray,
        y: np.ndarray,
        y_hat: np.ndarray
) -> None:
    if x.shape[1] != 2:
        print(f'Input data must have only two features, not {x.shape[1]}')
        return
    
    y = ensure_dims(y)
    
    y_hat = ensure_dims(y_hat)
    y_hat = convert_probability_to_boolean(y_hat)
    
    classes = np.unique(y)
    assert classes.shape[0] > 1, 'Need more than one class to predict'
    
    num_of_records, num_of_features = x.shape
    
    plt.figure(figsize=(12, 6))
    plt.title(f'Classification Predictions')
    colors = ['yellow', 'orange']
    # Ground-truth plots
    for c, label in zip(colors, classes):
        mask = np.argwhere(y.flatten() == label).flatten()
        plt.scatter(x[mask, 0], x[mask, 1], label=f'class {label}', alpha=0.25, color=c)
    
    # Prediction plots
    classes = np.unique(y_hat)
    both_y = np.concatenate((y_hat, y), axis=1)
    
    pred_list: Optional[np.ndarray] = None
    for label in classes:
        mask_correct = np.argwhere((both_y[:, 0] == label) & (both_y[:, 1] == label)).flatten()
        if pred_list is None:
            pred_list = mask_correct.copy()
        else:
            pred_list = np.unique(np.concatenate((pred_list, mask_correct), axis=None))
        plt.scatter(
            x[mask_correct, 0],
            x[mask_correct, 1],
            alpha=0.5,
            color='green',
            marker='.'
        )
    wrong_predictions = set(range(num_of_records)).difference(set(pred_list))
    wrong_predictions = np.asarray(list(wrong_predictions), dtype=np.int)
    plt.scatter(x[wrong_predictions, 0], x[wrong_predictions, 1],
                label=f'Wrong Predictions',
                alpha=0.5,
                color='red',
                marker='x'
                )
    
    plt.legend()
    plt.show()


def gen_regression_data(
        n_features: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = make_regression(
        n_samples=100,
        n_features=n_features,
        n_informative=n_features,
        random_state=9999
    )
    return x, y


def gen_classification_data(
        n_features: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=100,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        random_state=9999
    )
    return x, y
