import numpy as np
from typing import Tuple


def logistic_step(x: float, a: float) -> float:
    """Jeden krok logistickej mapy: x_{n+1} = a * x_n * (1 - x_n)."""
    # vzorec presne podla zadania
    return a * x * (1.0 - x)


def iterate_logistic(a: float, x0: float = 0.5,
                     transient: int = 200, samples: int = 100) -> np.ndarray:
    """Spusti logisticku mapu pre dany parameter a.

    Najprv vyhodi prvych `transient` iteracii (necha system ustalit sa),
    potom ulozi `samples` dalsich hodnot - tie tvoria stlpec v bifurcation.
    """
    x = x0
    # transient - zahrievaci beh, hodnoty zahodime
    for _ in range(transient):
        x = logistic_step(x, a)

    # zaznam ustalenych hodnot
    out = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        x = logistic_step(x, a)
        out[i] = x
    return out


def bifurcation_data(a_min: float = 0.0, a_max: float = 4.0, a_count: int = 1000,
                     x0: float = 0.5, transient: int = 200,
                     samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Vygeneruj data pre bifurcation diagram.

    Vrati (a_flat, x_flat) - dva 1D pole rovnakej dlzky:
    pre kazdy parameter a sa zaznamena `samples` ustalenych hodnot x.
    Vhodne na priamy plt.scatter / plt.plot s '.' markerom.
    """
    a_values = np.linspace(a_min, a_max, a_count)

    a_flat = np.repeat(a_values, samples)
    x_flat = np.empty(a_count * samples, dtype=np.float64)

    for i, a in enumerate(a_values):
        x_flat[i * samples:(i + 1) * samples] = iterate_logistic(a, x0, transient, samples)

    return a_flat, x_flat
