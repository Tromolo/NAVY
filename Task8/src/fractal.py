import numpy as np


def mandelbrot(xmin: float, xmax: float, ymin: float, ymax: float,
               width: int, height: int, max_iter: int = 256) -> np.ndarray:
    # rovnomerne rozdelenie rozsahu na width/height bodov - osi komplexnej roviny
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # kombinacia X a Y do 2D mrizky komplexnych cisel c = x + yi
    # kazdy pixel = jedno c
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    # z0 = 0 pre kazdy pixel
    Z = np.zeros_like(C)
    # pole - ukladame kolko iteracii bod prezil nez eskapoval
    iterations = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        # body este neeskapovali - |z| <= m, kde m = 2
        mask = np.abs(Z) <= 2
        # vzorec zn+1 = zn^2 + c, aplikujeme len na neeskapovane body
        Z[mask] = Z[mask] ** 2 + C[mask]
        # aktualizuj cislo iteracie - body v mnozine dosiahnu max_iter (cierne),
        iterations[mask] = i

    return iterations


def julia(xmin: float, xmax: float, ymin: float, ymax: float,
          width: int, height: int, c: complex = -0.7 + 0.27015j,
          max_iter: int = 256) -> np.ndarray:
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # pixel pozicia je pociatocna hodnota z0
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    iterations = np.zeros(Z.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        # vzorec zn+1 = zn^2 + c, ale c je fixna konstanta
        Z[mask] = Z[mask] ** 2 + c
        iterations[mask] = i

    return iterations
