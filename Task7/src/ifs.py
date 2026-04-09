import numpy as np

MODEL_1 = [
    # 1. transformacia
    {
        "matrix": np.array([[0.00, 0.00, 0.01],
                            [0.00, 0.26, 0.00],
                            [0.00, 0.00, 0.05]]),
        "offset": np.array([0.00, 0.00, 0.00]),
    },
    # 2. transformacia
    {
        "matrix": np.array([[0.20, -0.26, -0.01],
                            [0.23,  0.22, -0.07],
                            [0.07,  0.00,  0.24]]),
        "offset": np.array([0.00, 0.80, 0.00]),
    },
    # 3. transformacia
    {
        "matrix": np.array([[-0.25, 0.28, 0.01],
                            [ 0.26, 0.24, -0.07],
                            [ 0.07, 0.00,  0.24]]),
        "offset": np.array([0.00, 0.22, 0.00]),
    },
    # 4. transformacia
    {
        "matrix": np.array([[ 0.85, 0.04, -0.01],
                            [-0.04, 0.85,  0.09],
                            [ 0.00, 0.08,  0.84]]),
        "offset": np.array([0.00, 0.80, 0.00]),
    },
]

MODEL_2 = [
    # 1. transformacia
    {
        "matrix": np.array([[0.05, 0.00, 0.00],
                            [0.00, 0.60, 0.00],
                            [0.00, 0.00, 0.05]]),
        "offset": np.array([0.00, 0.00, 0.00]),
    },
    # 2. transformacia
    {
        "matrix": np.array([[ 0.45, -0.22,  0.22],
                            [ 0.22,  0.45,  0.22],
                            [-0.22,  0.22, -0.45]]),
        "offset": np.array([0.00, 1.00, 0.00]),
    },
    # 3. transformacia
    {
        "matrix": np.array([[-0.45,  0.22, -0.22],
                            [ 0.22,  0.45,  0.22],
                            [ 0.22, -0.22,  0.45]]),
        "offset": np.array([0.00, 1.25, 0.00]),
    },
    # 4. transformacia
    {
        "matrix": np.array([[ 0.49, -0.08,  0.08],
                            [ 0.08,  0.49,  0.08],
                            [ 0.08, -0.08,  0.49]]),
        "offset": np.array([0.00, 2.00, 0.00]),
    },
]


def generate_ifs(model: list[dict], n_iterations: int = 5000,
                 seed: int | None = None) -> np.ndarray:
    """
    Generuj 3D fraktal pomocou IFS (Iterated Function System).

    V kazdej iteracii sa nahodne vyberie jedna z transformacii
    s rovnakou pravdepodobnostou (p = 0.25) a aplikuje sa na aktualny bod.
    Kazdy vygenerovany bod sa ulozi do historie.

    Parametre:
        model: zoznam 4 transformacii (kazda ma 'matrix' a 'offset')
        n_iterations: pocet iteracii (pocet bodov ktore sa vygeneruju)
        seed: seed pre generator nahodnych cisel (reprodukovatelnost)

    Vracia: numpy pole tvaru (n_iterations, 3) s x, y, z suradnicami
    """
    rng = np.random.default_rng(seed)

    # Pociatocny bod
    point = np.array([0.0, 0.0, 0.0])

    # Historia vsetkych bodov pre vizualizaciu
    history = np.zeros((n_iterations, 3))

    n_transforms = len(model)

    probabilities = [1.0 / n_transforms] * n_transforms  # [0.25, 0.25, 0.25, 0.25]

    for i in range(n_iterations):
        # Nahodne vyber transformaciu podla pravdepodobnosti p = 0.25
        idx = rng.choice(n_transforms, p=probabilities)
        transform = model[idx]

        # Aplikuj afinnu transformaciu: w(x) = A * x + t
        point = transform["matrix"] @ point + transform["offset"]

        # Uloz bod do historie
        history[i] = point

    return history
