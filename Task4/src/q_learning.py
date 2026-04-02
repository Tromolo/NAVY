import numpy as np
from .environment import ACTIONS


class QLearning:
    """
    Q-learning agent pre gridove prostredie.

    Q-learning je off-policy algoritmus posilnovaneho ucenia (reinforcement learning),
    ktory sa uci Q-tabulku - mapovanie (stav, akcia) -> ocakavana odmena.
    Pouziva epsilon-greedy strategiu na vyvazenie prieskumu a vyuzivania.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        learning_rate: float = 0.1,   # alfa - rychlost ucenia (ako moc sa aktualizuje Q-hodnota)
        discount: float = 0.9,         # gamma - diskontny faktor (dolezitost buducich odmien)
        epsilon: float = 1.0,          # pociatocna pravdepodobnost nahodnej akcie (exploracie)
        epsilon_decay: float = 0.995,  # koeficient poklesu epsilonu po kazdej epizode
        epsilon_min: float = 0.01,     # minimalna hodnota epsilon (vzdy trochu exploruje)
    ):
        self.rows = rows
        self.cols = cols
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = len(ACTIONS)

        # Q-tabulka: 3D pole [riadok, stlpec, akcia] -> Q-hodnota
        # Pre kazde policko a kazdu akciu uklada ocakavanu kumulativnu odmenu
        self.q_table = np.zeros((rows, cols, self.n_actions))
        # Pamatova matica - pocitadlo navstev kazdeho policka pocas trenovania
        self.visit_counts = np.zeros((rows, cols), dtype=int)

    def choose_action(self, state: tuple[int, int]) -> int:
        """
        Epsilon-greedy vyber akcie.

        S pravdepodobnostou epsilon zvoli nahodnu akciu (exploracia/prieskum),
        inak zvoli akciu s najvyssou Q-hodnotou (exploitacia/vyuzivanie).
        """
        if np.random.random() < self.epsilon:
            # Nahodna akcia - prieskum noveho prostredia
            return np.random.randint(self.n_actions)
        # Greedy akcia - vyber najlepsiu podla Q-tabulky
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        """
        Aktualizacia Q-hodnoty podla Q-learning pravidla (Bellmanova rovnica):
        Q(s,a) = Q(s,a) + lr * (target - Q(s,a))

        kde target = odmena + gamma * max(Q(s', a'))  pre nekoncovy stav
            target = odmena                           pre koncovy stav
        """
        r, c = state
        nr, nc = next_state

        if done:
            # Koncovy stav - ziadna buduca odmena
            target = reward
        else:
            # Nekoncovy stav - odmena + diskontovana najlepsia buduca Q-hodnota
            target = reward + self.discount * np.max(self.q_table[nr, nc])

        # Aktualizuj Q-hodnotu smerom k targetu (posun o learning_rate)
        self.q_table[r, c, action] += self.lr * (target - self.q_table[r, c, action])

    def decay_epsilon(self) -> None:
        """Postupne znizuje epsilon - agent zacne viac exploitovat co sa naucil."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, n_episodes: int = 1000, max_steps: int = 200) -> list[dict]:
        """
        Natrenuje agenta na danom prostredi cez opakovane epizody.

        Kazda epizoda: mys zacina na startovej pozicii a pokusa sa najst syr.
        Postupne sa uci lepsiu strategiu aktualizaciou Q-tabulky.

        Vracia historiu epizod: [{"reward": ..., "steps": ..., "found_cheese": ...}]
        """
        # Reset Q-tabulky a statistik na zaciatku trenovania
        self.q_table[:] = 0
        self.visit_counts[:] = 0
        self.epsilon = 1.0  # zacni s plnou exploraciou
        history = []

        start = env.reset_mouse()

        for ep in range(n_episodes):
            state = start  # kazda epizoda zacina od startu
            total_reward = 0.0
            found_cheese = False

            for step in range(max_steps):
                # Zaznamenaj navstevu policka
                self.visit_counts[state[0], state[1]] += 1
                # Vyber akciu (epsilon-greedy)
                action = self.choose_action(state)
                # Vykonaj akciu v prostredi
                next_state, reward, done = env.step(state, action)
                # Aktualizuj Q-tabulku na zaklade skusenosti
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    # Epizoda skoncila - syr najdeny alebo pasticka
                    found_cheese = reward > 0
                    break

            # Po kazdej epizode zniz epsilon (menej nahodnych akcii)
            self.decay_epsilon()
            history.append({
                "reward": total_reward,
                "steps": step + 1,
                "found_cheese": found_cheese,
            })

        return history

    def get_best_path(self, env, max_steps: int = 200) -> list[tuple[int, int]]:
        """
        Vrati najlepsiu cestu podla naucnej Q-tabulky.
        Pouziva cisto greedy strategiu (vzdy najlepsia akcia, ziadna nahodnost).
        """
        state = env.reset_mouse()
        path = [state]

        for _ in range(max_steps):
            r, c = state
            # Vzdy vyber akciu s najvyssou Q-hodnotou
            action = int(np.argmax(self.q_table[r, c]))
            next_state, _, done = env.step(state, action)

            if next_state == state:
                # Agent sa zasekol (narazil do steny) - ukonci
                break
            path.append(next_state)
            state = next_state

            if done:
                break

        return path

    def get_policy_arrows(self) -> np.ndarray:
        """Vracia maticu najlepsich akcii (indexy) pre kazde policko."""
        return np.argmax(self.q_table, axis=2)
