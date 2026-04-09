import math

PRESETS = {
    1: {
        "name": "Koch Island",
        "axiom": "F+F+F+F",
        "rule": "F+F-F-FF+F+F-F",
        "angle": 90,
    },
    2: {
        "name": "Sierpinski Triangle",
        "axiom": "F++F++F",
        "rule": "F+F--F+F",
        "angle": 60,
    },
    3: {
        "name": "Fractal Plant",
        "axiom": "F",
        "rule": "F[+F]F[-F]F",
        "angle": math.degrees(math.pi / 7),
    },
    4: {
        "name": "Fractal Tree",
        "axiom": "F",
        "rule": "FF+[+F-F-F]-[-F+F+F]",
        "angle": math.degrees(math.pi / 8),
    },
}


def apply_rules(axiom: str, rule: str, n: int) -> str:
    """
    Aplikuj pravidlo prepisovania n-krat (nesting).

    Pre kazdu iteraciu nahrad kazdy vyskyt 'F' v retazci za dane pravidlo.
    Ostatne znaky (+, -, [, ], b) zostanu nezmenene.

    Parametre:
        axiom: pociatocny retazec (napr. "F+F")
        rule: pravidlo prepisovania pre F (napr. "F+F-F")
        n: pocet iteracii (uroven vnorenia/nesting)

    Vracia: vysledny retazec po n iteraciach
    """
    result = axiom
    for _ in range(n):
        new_result = []
        for ch in result:
            if ch == "F":
                new_result.append(rule)
            else:
                new_result.append(ch)
        result = "".join(new_result)
    return result


def interpret(instructions: str, step_length: float, angle_deg: float,
              start_x: float = 0.0, start_y: float = 0.0,
              start_angle_deg: float = 0.0) -> list[tuple[float, float, float, float, bool]]:
    """
    Interpretuj L-system retazec ako korytnaci grafiku (turtle graphics).

    Symboly:
        F  - posun dopredu o step_length (zanechat stopu)
        b  - posun dopredu bez stopy
        +  - otocenie vpravo o angle_deg
        -  - otocenie vlavo o angle_deg
        [  - uloz aktualnu poziciu a uhol na zasobnik
        ]  - obnov poziciu a uhol zo zasobnika

    Parametre:
        instructions: L-system retazec na interpretaciu
        step_length: dlzka jedneho kroku (F)
        angle_deg: uhol otacania v stupnoch
        start_x, start_y: pociatocna pozicia
        start_angle_deg: pociatocny uhol v stupnoch (0 = doprava)

    Vracia: zoznam useciek [(x1, y1, x2, y2, draw)] kde draw urcuje ci kreslit
    """
    # Aktualny stav korytnacky
    x = start_x
    y = start_y
    angle = math.radians(start_angle_deg)
    angle_rad = math.radians(angle_deg)

    # Zasobnik pre ukladanie pozicii pri [ a ]
    stack: list[tuple[float, float, float]] = []

    # Zoznam vsetkych useciek na vykreslenie
    segments: list[tuple[float, float, float, float, bool]] = []

    for ch in instructions:
        if ch == "F":
            # Posun dopredu so stopou
            nx = x + step_length * math.cos(angle)
            ny = y + step_length * math.sin(angle)
            segments.append((x, y, nx, ny, True))
            x, y = nx, ny
        elif ch == "b":
            # Posun dopredu bez stopy
            nx = x + step_length * math.cos(angle)
            ny = y + step_length * math.sin(angle)
            segments.append((x, y, nx, ny, False))
            x, y = nx, ny
        elif ch == "+":
            # Otocenie vpravo (kladny uhol = v smere hodinovych ruciciek)
            angle -= angle_rad
        elif ch == "-":
            # Otocenie vlavo (zaporny uhol = proti smeru hodinovych ruciciek)
            angle += angle_rad
        elif ch == "[":
            # Uloz aktualnu poziciu a uhol na zasobnik (checkpoint)
            stack.append((x, y, angle))
        elif ch == "]":
            # Obnov poslednu ulozenu poziciu a uhol
            if stack:
                x, y, angle = stack.pop()

    return segments
