"""
hopfield.py – Hopfieldova siet na ukladanie a obnovu vzorov.

Ucenie:   Hebbovo pravidlo  W += outer(p, p),  diag(W) = 0
Obnova:   synchronna (vsetky neurony naraz)
          asynchronna (neurony po jednom v nahodnom poradi)
"""

import numpy as np


class HopfieldNetwork:
    """Hopfieldova siet s binarnym stavom (+1 / -1)."""

    def __init__(self, n_neurons: int) -> None:
        self.n = n_neurons
        self.W = np.zeros((self.n, self.n), dtype=float)
        self.stored_patterns: list[np.ndarray] = []

    def add_pattern(self, pattern: np.ndarray) -> None:
        """Ulozi novy vzor (+1 / -1) a aktualizuje vahy (Hebbovo pravidlo)."""
        if pattern.shape != (self.n,):
            raise ValueError("Pattern has wrong size")
        self.stored_patterns.append(pattern.copy())
        self.W += np.outer(pattern, pattern) # vonkajsi produkt vzoru s jeho transponovanym vzorom
        np.fill_diagonal(self.W, 0.0) # vyplnime diagonalu nulami

    def _sync_update(self, state: np.ndarray) -> np.ndarray:
        """Jeden synchronny krok – aktualizacia vsetkych neuronov naraz."""
        net = self.W @ state # vypocita aktivacie pre vsetky neurony naraz
        return np.where(net >= 0, 1.0, -1.0) 

    def recover_sync(self, state: np.ndarray, max_iters: int = 20) -> np.ndarray:
        """Synchronna obnova – vsetky neurony naraz az do konvergencie."""
        current = state.copy()
        for _ in range(max_iters):
            new_state = self._sync_update(current)
            if np.array_equal(new_state, current):
                break
            current = new_state
        return current

    def recover_async(self, state: np.ndarray, max_iters: int = 200) -> np.ndarray:
        """Asynchronna obnova – neurony aktualizovane po jednom v nahodnom poradi."""
        current = state.copy()
        rng = np.random.default_rng()

        for _ in range(max_iters):
            i = int(rng.integers(0, self.n))
            net_i = float(self.W[i, :] @ current) # vypocita aktivacie pre dany neuron
            current[i] = 1.0 if net_i >= 0 else -1.0 
        return current
