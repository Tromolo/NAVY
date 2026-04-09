import matplotlib.pyplot as plt
from .ifs import MODEL_1, MODEL_2, generate_ifs


def plot_model(model, title: str, n_iterations: int = 5000) -> None:
    # Vygeneruj body fraktalu
    points = generate_ifs(model, n_iterations=n_iterations)

    # Vytvor 3D graf
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # scatter plot
    ax.scatter(
        points[:, 0],  # X suradnice
        points[:, 1],  # Y suradnice
        points[:, 2],  # Z suradnice
        s=1,           # velkost bodov
        c="black",     # farba bodov
        alpha=0.6,     # priehladnost
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def run_gui(n_iterations: int = 5000) -> None:
    plot_model(MODEL_1, "First model", n_iterations)
    plot_model(MODEL_2, "Second model", n_iterations)

    plt.show()
