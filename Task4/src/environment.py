import numpy as np

# Konstanty reprezentujuce typy policok v gride
EMPTY = 0    # prazdne policko - mys sa moze volne pohybovat
WALL = 1     # stena - nepriechodna prekazka
TRAP = 2     # pasticka - hra konci s velkou negativnou odmenou
CHEESE = 3   # syr - ciel hry, velka pozitivna odmena
MOUSE = 4    # pozicia mysi - startovaci bod agenta

# Definicia moznych akcii ako zmeny suradnic (riadok, stlpec)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # hore, dole, vlavo, vpravo
ACTION_NAMES = ["Hore", "Dole", "Vlavo", "Vpravo"]


class GridWorld:
    """
    Gridove prostredie pre hru 'Najdi syr'.

    Reprezentuje 2D mriezku kde sa mys (agent) snazi najst syr (ciel).
    Prostredie obsahuje steny (prekazky) a pasticky (nebezpecenstvo).
    Implementuje rozhranie step() pre interakciu s Q-learning agentom.
    """

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        # Inicializacia prazdneho gridu ako 2D numpy pole
        self.grid = np.zeros((rows, cols), dtype=int)
        self.mouse_pos: tuple[int, int] | None = None

    def place(self, row: int, col: int, cell_type: int) -> None:
        """Umiestni objekt (stenu, pasticku, syr, mys) na dane policko."""
        if cell_type == MOUSE:
            # Odstran staru poziciu mysi - mys moze byt len jedna
            if self.mouse_pos is not None:
                old_r, old_c = self.mouse_pos
                self.grid[old_r, old_c] = EMPTY
            self.mouse_pos = (row, col)
        self.grid[row, col] = cell_type

    def clear(self) -> None:
        """Vymaze cely grid - nastavi vsetky policka na EMPTY."""
        self.grid[:] = EMPTY
        self.mouse_pos = None

    def reset_mouse(self) -> tuple[int, int]:
        """Vrati poziciu mysi - pouziva sa na zaciatku kazdej epizody."""
        if self.mouse_pos is None:
            raise ValueError("Mys nie je umiestnena na gride.")
        return self.mouse_pos

    def step(self, state: tuple[int, int], action_idx: int) -> tuple[tuple[int, int], float, bool]:
        """
        Vykona akciu z daneho stavu. Toto je hlavna funkcia prostredia.

        Parametre:
            state: aktualna pozicia (riadok, stlpec)
            action_idx: index akcie (0=hore, 1=dole, 2=vlavo, 3=vpravo)

        Vracia: (novy_stav, odmena, koniec_hry)
            - odmena -1 za kazdy krok (motivuje agenta najst najkratsiu cestu)
            - odmena -100 za padnutie do pasticky
            - odmena +100 za najdenie syra
        """
        # Vypocitaj novu poziciu podla zvolenej akcie
        dr, dc = ACTIONS[action_idx]
        nr, nc = state[0] + dr, state[1] + dc

        # Mimo grid alebo stena -> zostan na mieste, mala penalizacia
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return state, -1.0, False
        if self.grid[nr, nc] == WALL:
            return state, -1.0, False

        # Pasticka -> velka penalizacia, koniec epizody
        if self.grid[nr, nc] == TRAP:
            return (nr, nc), -100.0, True

        # Syr -> velka odmena, koniec epizody (uspech)
        if self.grid[nr, nc] == CHEESE:
            return (nr, nc), 100.0, True

        # Prazdne policko -> posun sa tam, mala penalizacia za krok
        return (nr, nc), -1.0, False

    def has_cheese(self) -> bool:
        """Skontroluje ci je na gride umiestneny aspon jeden syr."""
        return np.any(self.grid == CHEESE)
