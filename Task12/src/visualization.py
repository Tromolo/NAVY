import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

from .forest_fire import create_grid, step, EMPTY, TREE, BURNING, BURNT


# farby podla zadania - hneda=empty, zelena=strom, oranzova=ohen, cierna=zhoreny
COLORS = ["#6b4a2b", "#2e8b3d", "#ff8c1a", "#111111"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm([EMPTY - 0.5, TREE - 0.5, BURNING - 0.5, BURNT - 0.5, BURNT + 0.5], CMAP.N)


def animate_forest_fire(size: int = 100,
                        p: float = 0.05,
                        f: float = 0.001,
                        density: float = 0.5,
                        neighborhood: str = "von_neumann",
                        interval_ms: int = 80,
                        seed: int | None = None) -> FuncAnimation:
    """Nekonecna animacia forest-fire CA - kazdy frame = jeden krok simulacie."""
    rng = np.random.default_rng(seed)
    grid = create_grid(size, density, rng)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title("Forest Fire")
    ax.set_title("Forest Fire")

    img = ax.imshow(grid, cmap=CMAP, norm=NORM, origin="lower", interpolation="nearest")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)

    state = {"grid": grid}

    def update(_frame):
        state["grid"] = step(state["grid"], p, f, rng, neighborhood)
        img.set_data(state["grid"])
        return (img,)

    # frames=None => generator bezi donekonecna; cache_frame_data vypnute
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=True,
                         cache_frame_data=False)
    fig._anim = anim  # referencia inak GC zabije animaciu
    return anim


def run_gui() -> None:
    """Spusti animaciu s defaultnymi parametrami zo zadania."""
    animate_forest_fire(
        size=100,
        p=0.05,
        f=0.001,
        density=0.5,
        neighborhood="von_neumann",
        interval_ms=80,
    )
    plt.show()
