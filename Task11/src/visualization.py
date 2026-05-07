import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .pendulum import simulate


DEFAULT_THETA1 = 2.0 * np.pi / 6.0
DEFAULT_THETA2 = 5.0 * np.pi / 8.0


def plot_trajectories(sim: dict) -> None:
    """Staticky graf - uhly theta1, theta2 v case + trajektoria m2 v xy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].plot(sim["t"], sim["theta1"], label=r"$\theta_1$", color="tab:red")
    axes[0].plot(sim["t"], sim["theta2"], label=r"$\theta_2$", color="tab:blue")
    axes[0].set_title("Uhly v case")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("uhol [rad]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(sim["x2"], sim["y2"], color="tab:purple", linewidth=0.8)
    axes[1].set_title(r"Trajektoria $m_2$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Double pendulum - chaotic motion")
    fig.tight_layout()


def animate_pendulum(sim: dict, l1: float, l2: float,
                     trail_length: int = 200, interval_ms: int = 16):
    """Animacia kyvadla - tyc + dve gulicky + zanechany chvost m2."""
    x1 = sim["x1"]
    y1 = sim["y1"]
    x2 = sim["x2"]
    y2 = sim["y2"]

    fig, ax = plt.subplots(figsize=(8, 8))
    reach = (l1 + l2) * 1.1
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Double pendulum")
    ax.grid(alpha=0.3)

    # zaves (pivot)
    ax.plot(0, 0, "ks", markersize=8)

    # tyce kyvadla
    rod_line, = ax.plot([], [], "-", color="black", lw=2)
    # gulicky
    bob1, = ax.plot([], [], "o", color="tab:red", markersize=14)
    bob2, = ax.plot([], [], "o", color="tab:blue", markersize=14)
    # chvost trajektorie m2
    trail, = ax.plot([], [], "-", color="tab:purple", lw=1.0, alpha=0.7)
    # casovy text
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top")

    t = sim["t"]

    def init():
        rod_line.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return rod_line, bob1, bob2, trail, time_text

    def update(i):
        rod_line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        bob1.set_data([x1[i]], [y1[i]])
        bob2.set_data([x2[i]], [y2[i]])
        start = max(0, i - trail_length)
        trail.set_data(x2[start:i + 1], y2[start:i + 1])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return rod_line, bob1, bob2, trail, time_text

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init,
                         interval=interval_ms, blit=True, repeat=False)
    # ulozit referenciu - inak ju GC odstreli a animacia zamrzne
    fig._anim = anim
    return anim


def run_gui() -> None:
    """Spusti simulaciu a vykresli staticky graf + animaciu."""
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    sim = simulate(
        theta1_0=DEFAULT_THETA1,
        theta2_0=DEFAULT_THETA2,
        l1=l1, l2=l2, m1=m1, m2=m2,
        duration=25.0,
        fps=60,
    )

    plot_trajectories(sim)
    animate_pendulum(sim, l1, l2, trail_length=250, interval_ms=16)

    plt.show()
