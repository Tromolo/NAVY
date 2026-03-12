"""
visualization.py – Animovana vizualizacia trenovania XOR siete.

Jeden graf so svetlou temou:
  - Rozhodovacia plocha (heatmapa) sa vyvija po epochach
  - 4 XOR body vyfarbene podla skutocneho navestia
  - Info box s epochou, stratou a predikciami
  - Slider na vyber epochy + Play/Pause
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap


# ── Farby ──────────────────────────────────────────────────────────────────────
COLOR_0 = "#3498db"   # modra  – trieda 0
COLOR_1 = "#e74c3c"   # cervena – trieda 1

# Farebna mapa pre rozhodovaci povrch (modra → biela → cervena)
_CMAP = LinearSegmentedColormap.from_list(
    "xor_cmap", [COLOR_0, "#f0f0f0", COLOR_1], N=256,
)


def _build_grid(resolution: int = 150):
    """Vytvori mriežku bodov pre vizualizaciu rozhodovacej plochy."""
    margin = 0.3
    xx, yy = np.meshgrid(
        np.linspace(-margin, 1 + margin, resolution),
        np.linspace(-margin, 1 + margin, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _predict_with_snapshot(grid: np.ndarray, snap: dict) -> np.ndarray:
    """Vypocita predikcie siete s vahami z danej snimky."""
    w_h = snap["w_hidden"]
    b_h = snap["b_hidden"]
    w_o = snap["w_output"]
    b_o = snap["b_output"]

    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    z_h = grid @ w_h + b_h
    a_h = sig(z_h)
    z_o = a_h @ w_o + b_o
    return sig(z_o)


def run_animation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    interval_ms: int = 120,
) -> None:
    """Zobrazi animaciu trenovania XOR siete."""

    xx, yy, grid = _build_grid()
    n_snaps = len(model.history)

    # Predpocitaj rozhodovacie plochy pre vsetky snimky
    surfaces = []
    for snap in model.history:
        zz = _predict_with_snapshot(grid, snap).reshape(xx.shape)
        surfaces.append(zz)

    # ── Figura ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.22)

    fig.suptitle("XOR Neural Network", fontsize=14, fontweight="bold")

    # Heatmapa (pociatocna)
    img = ax.imshow(
        surfaces[0],
        extent=[-0.3, 1.3, -0.3, 1.3],
        origin="lower",
        cmap=_CMAP,
        vmin=0, vmax=1,
        aspect="auto",
        alpha=0.7,
    )
    plt.colorbar(img, ax=ax, label="Vystup siete", shrink=0.85)

    # XOR body
    for i in range(len(X)):
        color = COLOR_1 if y[i] == 1 else COLOR_0
        marker = "^" if y[i] == 1 else "o"
        ax.scatter(
            X[i, 0], X[i, 1],
            c=color, s=200, marker=marker,
            edgecolors="black", linewidths=1.5, zorder=5,
            label=f"XOR={int(y[i])}" if i < 2 else None,
        )

    # Popisky bodov
    for i in range(len(X)):
        ax.annotate(
            f"({int(X[i,0])},{int(X[i,1])})",
            (X[i, 0], X[i, 1]),
            textcoords="offset points",
            xytext=(12, 8), fontsize=9,
        )

    # Info text
    info_text = ax.text(
        0.98, 0.97, "",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Stav ───────────────────────────────────────────────────────────────
    state = {"idx": 0, "playing": False, "_updating": False}

    def draw_snap(idx: int):
        snap = model.history[idx]
        img.set_data(surfaces[idx])

        preds = snap["predictions"]
        pred_str = "  ".join(f"{p:.2f}" for p in preds)

        info_text.set_text(
            f"Epocha: {snap['epoch']}\n"
            f"Loss:   {snap['loss']:.4f}\n"
            f"Pred:   [{pred_str}]"
        )
        fig.canvas.draw_idle()

    def set_idx(idx: int):
        idx = int(np.clip(idx, 0, n_snaps - 1))
        state["idx"] = idx
        state["_updating"] = True
        slider.set_val(idx)
        state["_updating"] = False
        draw_snap(idx)

    # ── Slider + tlacidlo ──────────────────────────────────────────────────
    slider_ax = fig.add_axes([0.15, 0.08, 0.55, 0.03])
    slider = Slider(slider_ax, "Snimka", 0, n_snaps - 1, valinit=0, valstep=1)

    btn_ax = fig.add_axes([0.78, 0.07, 0.12, 0.05])
    btn_play = Button(btn_ax, "Play")

    def on_slider(val):
        if state["_updating"]:
            return
        state["idx"] = int(val)
        draw_snap(int(val))

    def on_play(event):
        if state["playing"]:
            state["playing"] = False
            btn_play.label.set_text("Play")
        else:
            if state["idx"] >= n_snaps - 1:
                state["idx"] = 0
            state["playing"] = True
            btn_play.label.set_text("Pause")

    def anim_step(frame):
        if not state["playing"]:
            return []
        nxt = state["idx"] + 1
        if nxt >= n_snaps:
            state["playing"] = False
            btn_play.label.set_text("Play")
            return []
        set_idx(nxt)
        return []

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)

    draw_snap(0)

    anim = animation.FuncAnimation(
        fig, anim_step, interval=interval_ms, blit=False, cache_frame_data=False
    )
    fig._anim = anim  # type: ignore[attr-defined]

    plt.show()
