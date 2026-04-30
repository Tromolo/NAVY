import matplotlib.pyplot as plt
import numpy as np

from .logistic import bifurcation_data
from .model import build_training_set, train_predictor, predict_grid


def plot_bifurcation_only(a_min: float = 0.0, a_max: float = 4.0,
                          a_count: int = 1000, samples: int = 100,
                          transient: int = 200) -> None:
    """Example 1 - cisty bifurcation diagram."""
    a_flat, x_flat = bifurcation_data(a_min, a_max, a_count,
                                      transient=transient, samples=samples)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(a_flat, x_flat, ",k", alpha=0.5)
    ax.set_title("Bifurcation diagram")
    ax.set_xlabel("a")
    ax.set_ylabel("x")
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(0.0, 1.05)


def plot_training_vs_prediction(a_min: float = 0.0, a_max: float = 4.0,
                                a_count: int = 1000, samples: int = 100,
                                transient: int = 200,
                                pred_count: int = 1000) -> None:
    """Example 2 - dva grafy vedla seba: trenovacie data vs predikcia siete."""
    # 1) trenovaci bifurcation diagram
    a_flat, x_flat = bifurcation_data(a_min, a_max, a_count,
                                      transient=transient, samples=samples)

    # 2) trenovanie na mensej vzorke (rychlejsie)
    X_train, y_train = build_training_set(a_min, a_max,
                                          a_count=min(a_count, 800),
                                          transient=transient, samples=30)
    model = train_predictor(X_train, y_train)

    # 3) predikcia na novej mrizke
    a_pred, x_pred = predict_grid(model, a_min, a_max, pred_count)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].plot(a_flat, x_flat, ",", color="tab:blue", alpha=0.6)
    axes[0].set_title("Training data")
    axes[0].set_xlim(a_min, a_max)
    axes[0].set_ylim(0.0, 1.05)

    axes[1].plot(a_pred, x_pred, ".", color="tab:blue", markersize=2)
    axes[1].set_title("Network prediction")
    axes[1].set_xlim(a_min, a_max)
    axes[1].set_ylim(0.0, 1.05)


def plot_overlay_prediction(a_min: float = 0.0, a_max: float = 4.0,
                            a_count: int = 1000, samples: int = 100,
                            transient: int = 200,
                            pred_count: int = 400) -> None:
    """Example 3 - bifurcation diagram + cervene predikovane body cez."""
    a_flat, x_flat = bifurcation_data(a_min, a_max, a_count,
                                      transient=transient, samples=samples)

    X_train, y_train = build_training_set(a_min, a_max,
                                          a_count=min(a_count, 800),
                                          transient=transient, samples=30)
    model = train_predictor(X_train, y_train)
    a_pred, x_pred = predict_grid(model, a_min, a_max, pred_count)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(a_flat, x_flat, ",k", alpha=0.5, label="Actual")
    ax.plot(a_pred, x_pred, ".", color="red", markersize=4, label="Predicted")
    ax.set_title("Bifurcation Diagram with Predictions")
    ax.set_xlabel("Parameter (a)")
    ax.set_ylabel("Population")
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper left")


def run_gui() -> None:
    """Vykresli vsetky tri obrazky podla zadania."""
    plot_bifurcation_only()
    plot_training_vs_prediction()
    plot_overlay_prediction()
    plt.show()
