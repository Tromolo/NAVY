import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button


# ── Farby
COLOR_ABOVE = "#e74c3c"   # cervena  – nad priamkou (+1)
COLOR_BELOW = "#3498db"   # modra    – pod priamkou (-1)
COLOR_ON    = "#2ecc71"   # zelena   – na priamke (0)
COLOR_BOUND = "#e67e22"   # oranzova – rozhodovacia hranica
COLOR_TRUE  = "#2c3e50"   # tmava    – skutocna priamka


def run_animation(
    xs: np.ndarray,
    ys: np.ndarray,
    true_labels: np.ndarray,
    binary_labels: np.ndarray,
    perceptron,
    slope: float = 3,
    intercept: float = 2,
    interval_ms: int = 300,
) -> None:
    """Zobrazi animaciu trenovania perceptronu v jednom grafe."""

    x_min = float(xs.min()) - 0.5
    x_max = float(xs.max()) + 0.5
    x_line = np.linspace(x_min, x_max, 300)

    # Predpocitaj hranice pre vsetky epochy
    boundary_ys = [
        perceptron.boundary_y(x_line, epoch=i)
        for i in range(len(perceptron.history))
    ]
    n_epochs = len(perceptron.history) - 1

    # ── Figura
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.22)

    fig.suptitle(
        f"Perceptron  —  y = {slope}x + {intercept}",
        fontsize=14, fontweight="bold",
    )

    # Body podla skutocneho navestia
    mask_above = true_labels == 1
    mask_below = true_labels == -1
    mask_on    = true_labels == 0

    if mask_above.any():
        ax.scatter(xs[mask_above], ys[mask_above], c=COLOR_ABOVE,
                   s=40, label="Nad (+1)", zorder=3, alpha=0.8)
    if mask_below.any():
        ax.scatter(xs[mask_below], ys[mask_below], c=COLOR_BELOW,
                   s=40, label="Pod (-1)", zorder=3, alpha=0.8)
    if mask_on.any():
        ax.scatter(xs[mask_on], ys[mask_on], c=COLOR_ON,
                   s=55, marker="*", label="Na priamke (0)", zorder=4, alpha=0.9)

    # Skutocna priamka
    ax.plot(x_line, slope * x_line + intercept,
            color=COLOR_TRUE, linewidth=1.5, linestyle="-",
            label="Skutocna priamka", zorder=2)

    # Rozhodovacia hranica (animovana)
    (bound_line,) = ax.plot(
        [], [], color=COLOR_BOUND, linewidth=2, linestyle="--",
        label="Rozhodovacia hranica", zorder=5,
    )

    # Info text
    info_text = ax.text(
        0.98, 0.97, "",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y-limity
    all_y = np.concatenate([ys, slope * x_line + intercept])
    margin = (all_y.max() - all_y.min()) * 0.1
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # ── Stav ───────────────────────────────────────────────────────────────
    state = {"epoch": 0, "playing": False, "_updating": False}

    def draw_epoch(epoch: int):
        by = boundary_ys[epoch]
        if by is not None:
            bound_line.set_data(x_line, by)
        else:
            bound_line.set_data([], [])

        snap = perceptron.history[epoch]
        info_text.set_text(
            f"Epocha: {epoch}/{n_epochs}\n"
            f"Acc:    {snap['accuracy']:.1%}"
        )
        fig.canvas.draw_idle()

    def set_epoch(epoch: int):
        epoch = int(np.clip(epoch, 0, n_epochs))
        state["epoch"] = epoch
        state["_updating"] = True
        slider.set_val(epoch)
        state["_updating"] = False
        draw_epoch(epoch)

    # ── Slider + tlacidlo ──────────────────────────────────────────────────
    slider_ax = fig.add_axes([0.15, 0.08, 0.55, 0.03])
    slider = Slider(slider_ax, "Epocha", 0, n_epochs, valinit=0, valstep=1)

    btn_ax = fig.add_axes([0.78, 0.07, 0.12, 0.05])
    btn_play = Button(btn_ax, "Play")

    def on_slider(val):
        if state["_updating"]:
            return
        state["epoch"] = int(val)
        draw_epoch(int(val))

    def on_play(event):
        if state["playing"]:
            state["playing"] = False
            btn_play.label.set_text("Play")
        else:
            if state["epoch"] >= n_epochs:
                state["epoch"] = 0
            state["playing"] = True
            btn_play.label.set_text("Pause")

    def anim_step(frame):
        if not state["playing"]:
            return []
        nxt = state["epoch"] + 1
        if nxt > n_epochs:
            state["playing"] = False
            btn_play.label.set_text("Play")
            return []
        set_epoch(nxt)
        return []

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)

    draw_epoch(0)

    anim = animation.FuncAnimation(
        fig, anim_step, interval=interval_ms, blit=False, cache_frame_data=False
    )
    fig._anim = anim  # type: ignore[attr-defined]

    plt.show()
