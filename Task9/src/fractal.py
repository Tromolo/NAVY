import random
from typing import List, Tuple


Point = Tuple[float, float]


def midpoint_displacement(start: Point, end: Point, iterations: int,
                          offset: float, roughness: float = 0.5) -> List[Point]:
    """Generuj 2D fraktalny terain pomocou midpoint displacement.

    Algoritmus:
    1. Zacni s priamou ciarou (start -> end).
    2. Najdi stred ciary, posun ho vertikalne o nahodnu hodnotu (+/- offset).
    3. Vznikli dve nove ciary -> zopakuj rekurzivne.
    4. Kazdou iteraciou zmensi offset o roughness.

    Vrati zoznam bodov [(x, y), ...] zoradenych podla x.
    """
    # krok 1: priama ciara (start, end)
    points: List[Point] = [start, end]
    current_offset = offset

    for _ in range(iterations):
        new_points: List[Point] = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            # stred ciary
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0

            # krok 2+3: posun stredu hore/dole (50/50) v ramci offsetu
            mid_y += random.uniform(-current_offset, current_offset)

            # krok 4: z 1 ciary vzniknu 2 (p1->mid, mid->p2)
            new_points.append(p1)
            new_points.append((mid_x, mid_y))

        new_points.append(points[-1])
        points = new_points

        # krok 5: dalsia iteracia s mensim offsetom = jemnejsie detaily
        current_offset *= roughness

    return points


def midpoint_displacement_steps(start: Point, end: Point, iterations: int,
                                offset: float, roughness: float = 0.5) -> List[List[Point]]:
    """Vrati zoznam stavov po kazdej iteracii (vratane pociatocnej priamky).

    Pouzitelne pre animaciu - postupne sa zobrazuje rastuci pocet bodov.
    """
    points: List[Point] = [start, end]
    states: List[List[Point]] = [list(points)]
    current_offset = offset

    for _ in range(iterations):
        new_points: List[Point] = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            mid_y += random.uniform(-current_offset, current_offset)
            new_points.append(p1)
            new_points.append((mid_x, mid_y))
        new_points.append(points[-1])
        points = new_points
        states.append(list(points))
        current_offset *= roughness

    return states


def generate_landscape(start: Point, end: Point, layers_config: List[dict]) -> List[List[Point]]:
    """Generuj viacvrstvovy terain (pozadie az popredie).

    layers_config je zoznam slovnikov - kazdy popisuje jednu vrstvu:
        - y_offset: posun zakladnej ciary (vyssi = nizsie pohorie)
        - iterations: pocet iteracii subdivisionu
        - offset: pociatocna velkost perturbacie
        - roughness: faktor zmensenia perturbacie kazdou iteraciou
        - seed: nahodny seed pre reprodukovatelnost (volitelne)
    """
    layers: List[List[Point]] = []
    for cfg in layers_config:
        if "seed" in cfg and cfg["seed"] is not None:
            random.seed(cfg["seed"])

        s = (start[0], start[1] + cfg.get("y_offset", 0.0))
        e = (end[0], end[1] + cfg.get("y_offset", 0.0))

        layer = midpoint_displacement(
            s, e,
            iterations=cfg["iterations"],
            offset=cfg["offset"],
            roughness=cfg.get("roughness", 0.5),
        )
        layers.append(layer)
    return layers
