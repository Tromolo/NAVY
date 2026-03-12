import numpy as np
from src.data_generator import generate_points, to_binary
from src.perceptron import Perceptron
from src.visualization import run_animation


def main():
    # 1. Vygeneruj body okolo priamky y = 3x + 2
    xs, ys, labels = generate_points(n=100, seed=42)

    # 2. Preveď ternárne návestia (-1, 0, +1) na binárne (-1, +1)
    binary = to_binary(labels)

    # 3. Zostav maticu príznakov (N × 2)
    X = np.column_stack([xs, ys])

    # 4. Natrénuj perceptrón
    perc = Perceptron(learning_rate=0.1, n_epochs=50)
    perc.fit(X, binary)

    print(f"Finálna presnosť: {perc.accuracy(X, binary):.1%}")

    # 5. Spusti interaktívnu animáciu
    run_animation(xs, ys, labels, binary, perc, slope=3, intercept=2)


if __name__ == "__main__":
    main()
