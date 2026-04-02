import numpy as np
import matplotlib.pyplot as plt


def plot_training(q_history: list[dict], nn_losses: list[float]) -> None:
    """
    Vizualizacia priebehu trenovania - 3 grafy:
    1. Q-learning skore (ako sa zlepsoval agent pocas epizod)
    2. Epsilon decay (pokles nahodnosti pocas ucenia)
    3. Neuronova siet loss (pokles chyby siete pocas trenovania)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Q-learning (supervisor) + Neuronova siet: CartPole",
                 fontsize=14, fontweight="bold")

    # Graf 1: Q-learning skore v case
    # Modra (priehladna) = skore jednotlivych epizod
    # Cervena = klzavy priemer za 100 epizod (ukazuje trend ucenia)
    # Zelena ciarkована = cielove skore 200
    episodes = [h["episode"] for h in q_history]
    scores = [h["score"] for h in q_history]
    avg_scores = [h["avg_score"] for h in q_history]

    ax1.plot(episodes, scores, alpha=0.2, color="#3498db", label="Skore")
    ax1.plot(episodes, avg_scores, color="#e74c3c", linewidth=2, label="Priemer (100 ep.)")
    ax1.axhline(y=200, color="#2ecc71", linestyle="--", label="Ciel (200)")
    ax1.set_ylabel("Skore")
    ax1.set_title("Q-learning trenovanie (supervisor)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Graf 2: Pokles epsilon v case
    # Na zaciatku epsilon=1 (100% nahodne akcie), postupne klesa
    # Agent tak prechadza od prieskumu k vyuzivaniu nauceneho
    epsilons = [h["epsilon"] for h in q_history]
    ax2.plot(episodes, epsilons, color="#9b59b6", linewidth=2)
    ax2.set_xlabel("Epizoda")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon decay")
    ax2.grid(True, alpha=0.3)

    # Graf 3: Cross-entropy loss neuronovej siete
    # Klesajuci loss znamena ze siet sa uci spravne napodobovat Q-agenta
    ax3.plot(range(1, len(nn_losses) + 1), nn_losses, color="#e67e22", linewidth=2)
    ax3.set_xlabel("Epocha")
    ax3.set_ylabel("Cross-entropy loss")
    ax3.set_title("Neuronova siet - trenovanie z Q-tabulky")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_visual(env, nn, n_episodes: int = 3) -> None:
    """
    Spusti vizualne epizody kde natrenkovana neuronova siet riadi CartPole.
    Zobrazuje sa render prostredia v realnom case.
    """
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            # Siet predpovie akciu na zaklade aktualneho stavu
            action = nn.predict_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Epizoda {ep+1}: {total_reward:.0f} krokov")
                break
