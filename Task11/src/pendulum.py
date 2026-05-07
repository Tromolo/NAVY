import numpy as np
from scipy.integrate import odeint
from typing import Tuple


G = 9.81  # gravitacne zrychlenie [m/s^2]


def get_derivative(state: np.ndarray, t: float,
                   l1: float, l2: float, m1: float, m2: float) -> Tuple[float, float, float, float]:
    """Pravá strana ODE pre double pendulum.

    Vstup state = [theta1, omega1, theta2, omega2]
    Vystup tuple (theta1_dot, omega1_dot, theta2_dot, omega2_dot)
        kde omega_dot = uhlove zrychlenie podla Euler-Lagrange.
    """
    theta1, omega1, theta2, omega2 = state
    dtheta = theta1 - theta2
    sin_d = np.sin(dtheta)
    cos_d = np.cos(dtheta)

    # menovatel pre obe rovnice (rovnaky tvar - lisi sa l1/l2)
    denom = m1 + m2 * sin_d * sin_d

    # zrychlenie 1. kyvadla (theta1_dot_dot)
    num1 = (m2 * G * np.sin(theta2) * cos_d
            - m2 * sin_d * (l1 * omega1 * omega1 * cos_d + l2 * omega2 * omega2)
            - (m1 + m2) * G * np.sin(theta1))
    theta1_ddot = num1 / (l1 * denom)

    # zrychlenie 2. kyvadla (theta2_dot_dot)
    num2 = ((m1 + m2) * (l1 * omega1 * omega1 * sin_d
                         - G * np.sin(theta2)
                         + G * np.sin(theta1) * cos_d)
            + m2 * l2 * omega2 * omega2 * sin_d * cos_d)
    theta2_ddot = num2 / (l2 * denom)

    # odeint chce derivacie state premennych
    return omega1, theta1_ddot, omega2, theta2_ddot


def simulate(theta1_0: float, theta2_0: float,
             l1: float = 1.0, l2: float = 1.0,
             m1: float = 1.0, m2: float = 1.0,
             omega1_0: float = 0.0, omega2_0: float = 0.0,
             duration: float = 20.0, fps: int = 60) -> dict:
    """Spusti simulaciu - vrati slovnik s casovym polom a poziciami.

    duration v sekundach, fps urcuje hustotu casovych krokov.
    Vracia: t, theta1, theta2, x1, y1, x2, y2 (vsetko np.ndarray).
    """
    n_steps = int(duration * fps) + 1
    t = np.linspace(0.0, duration, n_steps)

    state_0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=np.float64)

    od = odeint(get_derivative, state_0, t, args=(l1, l2, m1, m2))

    theta1 = od[:, 0]
    theta2 = od[:, 2]

    # pozicie podla vzorcov "Positions of pendula"
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return {
        "t": t,
        "theta1": theta1,
        "theta2": theta2,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
    }
