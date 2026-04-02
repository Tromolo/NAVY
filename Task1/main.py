import numpy as np
from src.data_generator import generate_points, to_binary
from src.perceptron import Perceptron
from src.visualization import run_animation


def main():
    # 1. Vygeneruj body okolo priamky y = 3x + 2
    xs, ys, labels = generate_points(n=100, seed=42)

    # 2. Preveď ternárne návestia (-1, 0, +1) na binárne (-1, +1)
    binary = to_binary(labels)

    # 3. Matica príznakov
    X_train = np.column_stack([xs, ys])

    # 4. Trénovanie perceptronu
    perc = Perceptron(learning_rate=0.1, n_epochs=50)
    perc.fit(X_train, binary)

    print(f"Presnosť na trénovacej sade: {perc.accuracy(X_train, binary):.1%}")

    # 5. Testovacie dáta
    xs_test, ys_test, labels_test = generate_points(n=100, seed=99)
    binary_test = to_binary(labels_test)
    X_test = np.column_stack([xs_test, ys_test])

    print(f"Presnosť na testovacej sade:  {perc.accuracy(X_test, binary_test):.1%}")

    # 6. Animácia
    run_animation(xs, ys, labels, binary, perc, slope=3, intercept=2,
                  xs_test=xs_test, ys_test=ys_test,
                  labels_test=labels_test, binary_test=binary_test)


if __name__ == "__main__":
    main()
