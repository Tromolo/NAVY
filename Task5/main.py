import gymnasium as gym
from src.q_learning import QLearningAgent
from src.neural_network import NeuralNetwork
from src.visualization import plot_training, run_visual


def main():
    """
    Hlavny pipeline: Q-learning sa nauci hrat CartPole,
    potom neuronova siet sa nauci napodobovat Q-learning (supervised learning).

    CartPole: ulohou je udrzat tycku na voziku v rovnovahe co najdlhsie.
    Stav: (pozicia vozika, rychlost, uhol tycky, uhlova rychlost)
    Akcie: tlacit vlavo (0) alebo vpravo (1)
    """

    # 1. Vytvor CartPole prostredie z kniznice Gymnasium
    env = gym.make("CartPole-v1")

    # 2. KROK 1: Natrenuj Q-learning agenta (sluzi ako "ucitel/supervisor")
    # Q-learning sa uci optimalnu strategiu priamo z interakcie s prostredim
    print("=" * 50)
    print("KROK 1: Trenujem Q-learning (supervisor)")
    print("=" * 50)

    q_agent = QLearningAgent(
        n_bins=12,            # pocet binov pre diskretizaciu spojiteho stavu
        learning_rate=0.1,    # rychlost ucenia
        discount=0.99,        # diskontny faktor - vysoka hodnota = dlhodoby pohlad
        epsilon=1.0,          # pociatocna miera nahodnych akcii
        epsilon_decay=0.995,  # pokles epsilonu po kazdej epizode
        epsilon_min=0.01,     # minimalny epsilon
    )
    q_history = q_agent.train(env, n_episodes=5000, max_steps=200)

    # 3. KROK 2: Natrenuj neuronovu siet z Q-tabulky (supervised learning)
    # Siet sa uci predpovedat spravnu akciu pre spojity stav
    # Vyhodou je ze siet generalizuje - funguje aj pre stavy ktore neboli v Q-tabulke
    print("\n" + "=" * 50)
    print("KROK 2: Trenujem neuronovu siet z Q-tabulky")
    print("=" * 50)

    nn = NeuralNetwork(input_size=4, hidden_size=32, output_size=2, learning_rate=0.01)
    # Vygeneruj 50000 stavov z Q-agenta a pouzij ich ako trenovacie data pre siet
    nn_losses = nn.train_from_q_table(q_agent, env, n_samples=50000, n_epochs=50)

    env.close()

    # 4. Zobraz grafy priebehu trenovania (skore, epsilon, loss)
    plot_training(q_history, nn_losses)

    # 5. Vizualna ukazka - neuronova siet riadi vozik v realnom case
    print("\n" + "=" * 50)
    print("Vizualizacia: Neuronova siet (naucena z Q-learningu)")
    print("=" * 50)
    env_vis = gym.make("CartPole-v1", render_mode="human")
    run_visual(env_vis, nn, n_episodes=3)
    env_vis.close()


if __name__ == "__main__":
    main()
