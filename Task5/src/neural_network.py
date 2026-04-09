import numpy as np


class NeuralNetwork:
    """
    Neuronova siet s 2 skrytymi vrstvami.

    Architektura: vstup(4) -> skryta1(32, ReLU) -> skryta2(16, ReLU) -> vystup(2, Softmax)

    Vstup:  spojity stav CartPole (pozicia, rychlost, uhol, uhlova rychlost)
    Vystup: pravdepodobnosti 2 akcii (vlavo, vpravo)

    Ucena supervisovane z Q-learning tabulky - Q-learning sluzi ako ucitel,
    siet sa uci napodobovat jeho rozhodnutia ale pracuje priamo so spojitym stavom.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32, output_size: int = 2,
                 learning_rate: float = 0.01):
        self.lr = learning_rate
        h2 = hidden_size // 2  # druha skryta vrstva ma polovicny pocet neuronov

        # Inicializacia vah pomocou He inicializacie (vhodna pre ReLU)
        # sqrt(2/n) zabezpecuje vhodnu variabilitu gradientov
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)       # biasy 1. vrstvy
        self.W2 = np.random.randn(hidden_size, h2) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(h2)                # biasy 2. vrstvy
        self.W3 = np.random.randn(h2, output_size) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_size)       # biasy vystupnej vrstvy

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU aktivacna funkcia: max(0, x) - jednoducha a efektivna nelinearita."""
        return np.maximum(0, x)

    def _relu_deriv(self, x: np.ndarray) -> np.ndarray:
        """Derivacia ReLU: 1 ak x > 0, inak 0. Pouziva sa pri spatnom prechode."""
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax: prevod logitov na pravdepodobnosti (sucet = 1). Odcitanie maxima zabranuje preteceniu."""
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Dopredny prechod (forward pass) cez vsetky vrstvy siete.

        vstup -> W1*x+b1 -> ReLU -> W2*a1+b2 -> ReLU -> W3*a2+b3 -> Softmax -> pravdepodobnosti
        Medzivysledky sa ukladaju pre spatny prechod (backpropagation).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)  # pridaj batch dimenziu ak chyba
        self._x = x                           # vstup (ulozeny pre backprop)
        self._z1 = x @ self.W1 + self.b1      # linearna transformacia 1. vrstvou
        self._a1 = self._relu(self._z1)        # aktivacia 1. vrstvy
        self._z2 = self._a1 @ self.W2 + self.b2  # linearna transformacia 2. vrstvou
        self._a2 = self._relu(self._z2)        # aktivacia 2. vrstvy
        self._z3 = self._a2 @ self.W3 + self.b3  # linearna transformacia vystupnou vrstvou
        self._out = self._softmax(self._z3)    # softmax -> pravdepodobnosti akcii
        return self._out

    def predict_action(self, state: np.ndarray) -> int:
        """Vrati najlepsiu akciu pre dany stav (akciu s najvyssou pravdepodobnostou)."""
        probs = self.forward(state)
        return int(np.argmax(probs[0]))

    def train_batch(self, states: np.ndarray, labels: np.ndarray) -> float:
        """
        Jeden trenovaci krok na mini-batchi pomocou backpropagation.

        1. Forward pass - vypocitaj predikcie
        2. Vypocitaj cross-entropy loss (meria rozdiel medzi predikciami a spravnymi akciami)
        3. Backward pass - vypocitaj gradienty vah (retazove pravidlo / chain rule)
        4. Aktualizuj vahy v smere zaporne gradientu (gradient descent)

        states: (N, 4) - spojite stavy
        labels: (N,) - spravne akcie z Q-tabulky (0 alebo 1)
        """
        n = len(states)
        probs = self.forward(states)  # forward pass

        # One-hot encoding cielovych akcii: [0,1] alebo [1,0]
        targets = np.zeros_like(probs)
        targets[np.arange(n), labels.astype(int)] = 1.0

        # Cross-entropy loss: -sum(target * log(predikcia))
        # Meria ako daleko su predikcie od spravnych odpovedi
        loss = -np.mean(np.sum(targets * np.log(probs + 1e-8), axis=1))

        # === Spatny prechod (backpropagation) ===
        # Gradient sa siri od vystupu smerom ku vstupu (retazove pravidlo)

        # Gradient vystupnej vrstvy (softmax + cross-entropy sa zjednodusi)
        d_z3 = (probs - targets) / n
        d_W3 = self._a2.T @ d_z3  # gradient vah 3. vrstvy
        d_b3 = d_z3.sum(axis=0)   # gradient biasov 3. vrstvy

        # Gradient 2. skrytej vrstvy
        d_z2 = (d_z3 @ self.W3.T) * self._relu_deriv(self._z2)
        d_W2 = self._a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        # Gradient 1. skrytej vrstvy
        d_z1 = (d_z2 @ self.W2.T) * self._relu_deriv(self._z1)
        d_W1 = self._x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # Gradient clipping - orez prilis velkych gradientov pre stabilne ucenie
        for g in [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]:
            np.clip(g, -5.0, 5.0, out=g)

        # Aktualizuj vahy a biasy - posun v smere zaporne gradientu
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3

        return loss

    def train_from_q_table(self, q_agent, env, n_samples: int = 50000,
                           n_epochs: int = 50, batch_size: int = 128) -> list[float]:
        """
        Natrenuj siet z Q-tabulky (supervised/supervisovane ucenie).

        Postup:
        1. Spusti Q-agenta (greedy) v prostredi a zbieraj pary (stav, akcia)
        2. Pouzij tieto pary ako trenovacie data pre neuronovu siet
        3. Siet sa uci predpovedat akciu pre lubovolny spojity stav

        Vyhoda oproti Q-tabulke: siet generalizuje na stavy ktore neboli
        v Q-tabulke (nediskretizuje, pracuje priamo so spojitymi hodnotami).
        """
        print("Generujem trenovacie data z Q-tabulky...")

        states = []
        actions = []

        # Zbieraj data z epizod - Q-agent hra greedy a zaznamenavame jeho rozhodnutia
        for _ in range(n_samples // 200 + 1):
            state, _ = env.reset()
            for _ in range(200):
                action = q_agent.get_action_greedy(state)  # najlepsia akcia podla Q-tabulky
                states.append(state.copy())   # uloz spojity stav
                actions.append(action)         # uloz akciu ktoru Q-agent zvolil
                next_state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                state = next_state

        # Orez na pozadovany pocet vzoriek
        states = np.array(states[:n_samples])
        actions = np.array(actions[:n_samples])

        print(f"Trenujem neuronovu siet na {len(states)} vzorkach...")
        losses = []

        for epoch in range(n_epochs):
            # Zamiesaj data - predchadza pretreningu na poradii dat
            idx = np.random.permutation(len(states))
            epoch_loss = 0
            n_batches = 0

            # Trenuj po mini-batchoch (128 vzoriek naraz)
            for i in range(0, len(states), batch_size):
                batch_idx = idx[i:i + batch_size]
                loss = self.train_batch(states[batch_idx], actions[batch_idx])
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            # Kazdych 10 epoch vypis progres a presnost
            if epoch % 10 == 0:
                preds = np.array([self.predict_action(s) for s in states[:1000]])
                acc = np.mean(preds == actions[:1000])  # % spravnych predikcii
                print(f"Epocha {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1%}")

        return losses
