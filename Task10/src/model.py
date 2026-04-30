import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import Tuple

from .logistic import iterate_logistic


def build_training_set(a_min: float = 0.0, a_max: float = 4.0, a_count: int = 1000,
                       x0: float = 0.5, transient: int = 200,
                       samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Postav trenovacie data pre neuronku.

    Vstup: parameter a (1 feature).
    Vystup: jedna ustalena hodnota x pre dane a (1 target).

    Pre kazde a vezmeme niekolko ustalenych hodnot - aby sa siet naucila
    aj chaoticke regiony (kde existuje viac moznych x pre dane a).
    """
    a_values = np.linspace(a_min, a_max, a_count)

    X_list = []
    y_list = []
    for a in a_values:
        # pre kazde a vezmeme `samples` ustalenych hodnot - v chaotickom
        # regione tak siet vidi cely rozptyl, nie len jednu hodnotu
        ys = iterate_logistic(a, x0, transient, samples)
        X_list.append(np.full(samples, a))
        y_list.append(ys)

    # X = parameter a (1 feature), y = ustalena hodnota x (1 target)
    X = np.concatenate(X_list).reshape(-1, 1)
    y = np.concatenate(y_list)
    return X, y


def train_predictor(X: np.ndarray, y: np.ndarray,
                    hidden_layer_sizes: Tuple[int, ...] = (64, 64, 32),
                    max_iter: int = 200,
                    random_state: int = 42) -> MLPRegressor:
    """Natrenuj MLPRegressor na predikciu x z parametra a."""
    # MLP - 3 skryte vrstvy (64,64,32), ReLU aktivacia, Adam optimizer
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X, y)
    return model


def predict_grid(model: MLPRegressor,
                 a_min: float = 0.0, a_max: float = 4.0,
                 a_count: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Predikuj x pre rovnomerne rozlozene a hodnoty."""
    a_values = np.linspace(a_min, a_max, a_count)
    X = a_values.reshape(-1, 1)
    y_pred = model.predict(X)
    return a_values, y_pred
