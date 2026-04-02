import numpy as np


class QLearningAgent:
    """
    Q-learning agent s diskretizovanymi stavmi pre CartPole.

    Problem: CartPole ma spojity stavovy priestor (4 realne cisla),
    ale Q-learning potrebuje diskretne stavy pre Q-tabulku.
    Riesenie: kazdy rozmer stavu sa rozdeli do binov (intervalov),
    cim vznikne diskretny stavovy priestor.

    Q-tabulka ma rozmer (n_bins^4 * 2) - pre kazdu kombinaciu
    diskretizovanych stavov a 2 akcie (vlavo/vpravo).
    """

    def __init__(
        self,
        n_bins: int = 12,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_bins = n_bins
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = 2  # vlavo / vpravo

        # Hranice binov pre diskretizaciu 4 rozmerov spojiteho stavu
        # Kazdy rozmer sa rozdeli na n_bins intervalov
        self.bins = [
            np.linspace(-2.4, 2.4, n_bins - 1),     # x - pozicia vozika na draze
            np.linspace(-3.0, 3.0, n_bins - 1),      # v - rychlost vozika
            np.linspace(-0.26, 0.26, n_bins - 1),    # theta - uhol naklonu tycky (~15°)
            np.linspace(-3.0, 3.0, n_bins - 1),      # w - uhlova rychlost tycky
        ]

        # Q-tabulka: 5D pole (bin_x, bin_v, bin_theta, bin_w, akcia)
        # Kazda bunka obsahuje Q-hodnotu = ocakavanu kumulativnu odmenu
        self.q_table = np.zeros((n_bins,) * 4 + (self.n_actions,))

    def discretize(self, state: np.ndarray) -> tuple:
        """
        Prevedie spojity stav na diskretny index.

        np.digitize najde do ktoreho binu patri hodnota kazdeho rozmeru.
        Napr. stav [0.5, -1.2, 0.05, 0.3] -> (7, 3, 7, 6)
        """
        return tuple(
            int(np.digitize(state[i], self.bins[i]))
            for i in range(4)
        )

    def choose_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy vyber akcie.
        S pravdepodobnostou epsilon nahodna akcia, inak najlepsia z Q-tabulky.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        d_state = self.discretize(state)
        return int(np.argmax(self.q_table[d_state]))

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update pravidlo (Bellmanova rovnica):
        Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        ds = self.discretize(state)
        dns = self.discretize(next_state)

        if done:
            target = reward  # koncovy stav - ziadna buduca odmena
        else:
            # odmena + diskontovana najlepsia buduca Q-hodnota
            target = reward + self.discount * np.max(self.q_table[dns])

        # Aktualizuj Q-hodnotu smerom k targetu
        self.q_table[ds + (action,)] += self.lr * (target - self.q_table[ds + (action,)])

    def decay_epsilon(self):
        """Zniz epsilon - postupne prechadza z prieskumu na vyuzivanie nauceneho."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, n_episodes: int = 5000, max_steps: int = 200) -> list[dict]:
        """
        Natrenuje Q-tabulku na CartPole cez opakovane epizody.

        V kazdej epizode agent interaguje s prostredim a aktualizuje Q-tabulku.
        Postupne sa uci udrzat tycku dlhsie (vyssie skore = viac krokov).
        Vracia historiu epizod s hodnotami skore a epsilon.
        """
        history = []
        best_avg = 0

        for ep in range(n_episodes):
            state, _ = env.reset()  # reset prostredia na zaciatok epizody
            total_reward = 0

            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # epizoda konci padom tycky alebo casovym limitom

                # Aktualizuj Q-tabulku na zaklade skusenosti
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.decay_epsilon()

            # Zaznamenaj statistiky epizody
            history.append({
                "episode": ep + 1,
                "score": total_reward,
                "epsilon": self.epsilon,
            })

            # Vypocitaj klzavy priemer skore za poslednych 100 epizod
            if len(history) >= 100:
                avg = np.mean([h["score"] for h in history[-100:]])
                history[-1]["avg_score"] = avg
                if avg > best_avg:
                    best_avg = avg
            else:
                history[-1]["avg_score"] = np.mean([h["score"] for h in history])

            # Vypisuj progres kazdych 500 epizod
            if ep % 500 == 0:
                avg = history[-1]["avg_score"]
                print(f"Ep {ep+1}/{n_episodes} | "
                      f"Skore: {total_reward:.0f} | "
                      f"Priemer(100): {avg:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        print(f"\nTrenovanie hotove. Najlepsi priemer(100): {best_avg:.1f}")
        return history

    def get_action_greedy(self, state: np.ndarray) -> int:
        """Vrati najlepsiu akciu bez nahodnosti (pouziva sa pri generovani dat pre siet)."""
        d_state = self.discretize(state)
        return int(np.argmax(self.q_table[d_state]))
