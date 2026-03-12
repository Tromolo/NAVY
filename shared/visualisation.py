import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_points_and_line(xs: np.ndarray, ys: np.ndarray, labels: np.ndarray, a: float, b: float):
    eps = 1e-6

    above_idx = labels > eps
    below_idx = labels < -eps
    on_idx = ~(above_idx | below_idx)

    plt.figure(figsize=(7, 5))
    plt.title("Linear function")

    x_line = np.linspace(xs.min() - 1, xs.max() + 1, 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, "k-", label=f"y = {a}x + {b}")

    plt.scatter(xs[above_idx], ys[above_idx], c="red", label="nad priamkou")
    plt.scatter(xs[below_idx], ys[below_idx], c="blue", label="pod priamkou")
    plt.scatter(xs[on_idx], ys[on_idx], c="green", label="na priamke", marker="x", s=80)

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def animate_perceptron_training(
    xs: np.ndarray,
    ys: np.ndarray,
    a: float,
    b: float,
    weight_history: list[np.ndarray],
    save_path: str = "perceptron_training.gif",
    interval_ms: int = 150,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Perceptron training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    ax.scatter(xs, ys, s=20, c="gray", alpha=0.5)

    x_min, x_max = xs.min() - 1, xs.max() + 1
    x_line = np.linspace(x_min, x_max, 100)

    true_line, = ax.plot(x_line, a * x_line + b, "k-", label="y = 3x + 2")
    perc_line, = ax.plot([], [], "r--", label="perceptron boundary")

    ax.legend()

    def init():
        perc_line.set_data([], [])
        return perc_line, true_line

    def update(frame: int):
        w = weight_history[frame]
        w0, w1, w2 = w
        if abs(w1) < 1e-8:
            y_vals = np.full_like(x_line, np.nan)
        else:
            y_vals = -(w0 * x_line + w2) / w1
        perc_line.set_data(x_line, y_vals)
        ax.set_title(f"Perceptron training – epoch {frame + 1}")
        return perc_line, true_line

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(weight_history),
        interval=interval_ms,
        blit=True,
    )

    anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
    plt.close(fig)

