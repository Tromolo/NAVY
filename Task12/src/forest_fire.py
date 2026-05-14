import numpy as np

EMPTY = 0    # prazdna pôda (hneda na obrazku)
TREE = 1     # zivy strom (zelena)
BURNING = 2  # horiaci strom (oranzova)
BURNT = 3    # zhoreny strom (cierna)


def create_grid(size: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Vytvori pociatocny les ako 2D mriezku.

    size      - rozmer mriezky (size x size buniek)
    density   - pravdepodobnost, ze konkretna bunka bude na zaciatku strom
    rng       - generator nahodnych cisiel (numpy)

    Logika:
      - vygenerujem nahodne cisla 0..1 pre kazdu bunku
      - ak je cislo < density => na danej bunke je strom (TREE)
      - inak je bunka prazdna (EMPTY)
    Vystup: 2D numpy pole s hodnotami {EMPTY, TREE}.
    """
    # rng.random(...) vrati pole cisel od 0 do 1 rovnakeho tvaru ako mriezka
    random_values = rng.random((size, size))

    # vyber stromu/prazdna
    grid = np.where(random_values < density, TREE, EMPTY)

    return grid.astype(np.int8)


def step(grid: np.ndarray, p: float, f: float,
         rng: np.random.Generator, neighborhood: str = "von_neumann") -> np.ndarray:
    """Vykona JEDEN krok forest-fire celulárneho automatu.

    Pravidla podla zadania:
      1) Empty alebo BURNT bunka sa s pravdepodobnostou p stane stromom,
         inak ostane prazdna.
      2) Strom sa zacne hoiret, ak je v jeho okoli aspon jeden horiaci sused.
      3) Ak strom nema horiaceho suseda, samovolne sa vznieti s pravdep. f
         (predstavuje uder blesku).
      4) Horiaci strom v dalsom kroku zhori -> stav BURNT.

    grid          - aktualny stav mriezky
    p             - pravdepodobnost narastenia noveho stromu
    f             - pravdepodobnost samovolneho vznietenia (blesk)
    rng           - generator nahodnych cisiel
    neighborhood  - 'von_neumann' (4 susedia) alebo 'moore' (8 susedov)

    Cela funkcia je VEKTORIZOVANA - ziadne for cykly cez bunky.
    Pracujeme s celym polom naraz cez numpy operacie => velmi rychle.
    """

    # vyrobim si masku horiacich buniek: 1 ak horí, 0 inak
    burning = (grid == BURNING).astype(np.int8)

    # np.roll(pole, posun, axis) posunie cele pole o 1 v danom smere.
    n_up    = np.roll(burning,  1, axis=0)
    n_down  = np.roll(burning, -1, axis=0)
    n_left  = np.roll(burning,  1, axis=1)
    n_right = np.roll(burning, -1, axis=1)

    # Suma povie pre kazdu bunku pocet horiacich von-Neumann susedov (0..4)
    neighbors = n_up + n_down + n_left + n_right

    # Ak chceme Moorovo okolie, pridame este 4 diagonalnych susedov
    if neighborhood == "moore":
        n_ul = np.roll(n_up,    1, axis=1)
        n_ur = np.roll(n_up,   -1, axis=1)
        n_dl = np.roll(n_down,  1, axis=1)
        n_dr = np.roll(n_down, -1, axis=1)
        neighbors = neighbors + n_ul + n_ur + n_dl + n_dr

    # Tym padom horny okraj by videl susedov zo spodneho okraja, co nechceme.
    # Preto na hranach odpocitam tych "falsovych" susedov co prisli z druhej strany.
    neighbors[0, :]  = neighbors[0, :]  - n_up[0, :]
    neighbors[-1, :] = neighbors[-1, :] - n_down[-1, :]
    neighbors[:, 0]  = neighbors[:, 0]  - n_left[:, 0] 
    neighbors[:, -1] = neighbors[:, -1] - n_right[:, -1] 

    new_grid = grid.copy()

    # PRAVIDLO 4: vsetko co teraz hori, v dalsom kroku zhori
    new_grid[grid == BURNING] = BURNT

    # PRAVIDLO 1: prazdne aj zhorene bunky mozu vyrast na novy strom
    # maska buniek, ktore mozu narast
    regrow_mask = (grid == EMPTY) | (grid == BURNT)
    # nahodne rozhodnutie pre kazdu bunku zvlast
    new_trees = rng.random(grid.shape) < p
    # tam kde regrow_mask AND new_trees -> strom
    new_grid[regrow_mask &  new_trees] = TREE
    # tam kde regrow_mask AND nie new_trees -> ostava empty
    new_grid[regrow_mask & ~new_trees] = EMPTY

    # PRAVIDLO 2: strom so susedom v plameni zacne hoiret
    tree_mask = (grid == TREE)
    catches = tree_mask & (neighbors > 0)  # strom co ma horiaceho suseda
    new_grid[catches] = BURNING

    # PRAVIDLO 3: strom bez horiaceho suseda sa moze sam zapalit (blesk)
    lightning = tree_mask & ~catches & (rng.random(grid.shape) < f)
    new_grid[lightning] = BURNING

    return new_grid
