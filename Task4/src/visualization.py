import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
from .environment import GridWorld, EMPTY, WALL, TRAP, CHEESE, MOUSE, ACTION_NAMES
from .q_learning import QLearning

# Definicia farieb pre jednotlive typy policok
COLOR_EMPTY = "#ffffff"   # biela - prazdne policko
COLOR_WALL = "#6c757d"    # siva - stena
COLOR_TRAP = "#e74c3c"    # cervena - pasticka
COLOR_CHEESE = "#f1c40f"  # zlta - syr (ciel)
COLOR_MOUSE = "#3498db"   # modra - mys (agent)
COLOR_PATH = "#2ecc71"    # zelena - najdena cesta
COLOR_GRID = "#cccccc"    # svetlosiva - mriezka

# Smery sipok pre vizualizaciu politiky (dx, dy) pre akcie: hore, dole, vlavo, vpravo
ARROW_DX = [0, 0, -0.3, 0.3]
ARROW_DY = [0.3, -0.3, 0, 0]


def run_gui(rows: int = 10, cols: int = 10) -> None:
    """
    Spusti interaktivne GUI pre hru 'Najdi syr'.

    Pouzivatel moze:
    - Kliknutim umiestnat mys, steny, pasticky a syr na grid
    - Spustit trenovanie Q-learning agenta
    - Vizualizovat najdenu cestu, Q-hodnoty a pamatovu maticu
    """

    # Vytvor prostredie (grid) a Q-learning agenta
    env = GridWorld(rows, cols)
    agent = QLearning(rows, cols)

    # ── Vytvorenie matplotlib figury s gridovym zobrazenim
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(right=0.68, bottom=0.05, top=0.95)  # miesto pre tlacidla vpravo

    fig.suptitle("Q-learning: Najdi syr", fontsize=14, fontweight="bold")

    # Stav aplikacie - uchovava aktualny nastroj, ci je agent natreneny, atd.
    state = {
        "tool": MOUSE,           # aktualny nastroj pre kreslenie (mys/stena/pasticka/syr)
        "trained": False,        # ci uz prebehlo trenovanie
        "show_q": False,         # ci sa zobrazuju Q-hodnoty
        "show_visits": False,    # ci sa zobrazuje pamatova matica
        "path_artists": [],      # graficke objekty cesty
        "arrow_artists": [],     # graficke objekty sipok
        "anim": None,            # referencia na animaciu (aby nebola zrusena GC)
        "info_text": None,       # textovy objekt pre informacne spravy
    }

    def cell_color(val: int) -> str:
        """Vrati farbu pozadia pre dany typ policka."""
        if val == WALL:
            return COLOR_WALL
        if val == TRAP:
            return COLOR_TRAP
        if val == CHEESE:
            return COLOR_CHEESE
        if val == MOUSE:
            return COLOR_MOUSE
        return COLOR_EMPTY

    def cell_symbol(val: int) -> str:
        """Vrati textovy symbol pre dany typ policka (zobrazeny v strede)."""
        if val == WALL:
            return "█"
        if val == TRAP:
            return "O"
        if val == CHEESE:
            return "C"
        if val == MOUSE:
            return "✱"
        return ""

    def draw_grid():
        """Prekresli cely grid - zakladne zobrazenie s polickami a symbolmi."""
        ax.clear()
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # riadok 0 je hore
        ax.set_xticks([])
        ax.set_yticks([])

        # Vykresli kazde policko ako farebny obdlznik so symbolom
        for r in range(rows):
            for c in range(cols):
                val = env.grid[r, c]
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.5, edgecolor=COLOR_GRID,
                    facecolor=cell_color(val),
                )
                ax.add_patch(rect)

                sym = cell_symbol(val)
                if sym:
                    ax.text(c + 0.5, r + 0.5, sym,
                            ha="center", va="center", fontsize=14, fontweight="bold")

        # Vycisti stare artefakty (cesta, sipky z predchadzajuceho behu)
        state["path_artists"] = []
        state["arrow_artists"] = []

        fig.canvas.draw_idle()

    def draw_q_values():
        """
        Zobraz max Q-hodnoty pre kazde policko ako teplotnu mapu.

        Cervena = nizka Q-hodnota (zle policko), zelena = vysoka (dobre policko).
        Na kazdом policku je sipka ukazujuca najlepsiu akciu (politiku).
        """
        ax.clear()
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Q-hodnoty (max cez akcie)", fontsize=11)

        # Najdi maximalne Q-hodnoty pre kazde policko (cez vsetky akcie)
        q_max = np.max(agent.q_table, axis=2)
        vmin, vmax = q_max.min(), q_max.max()
        if vmin == vmax:
            vmax = vmin + 1  # zabran deleniu nulou pri normalizacii

        for r in range(rows):
            for c in range(cols):
                val = env.grid[r, c]
                if val == WALL:
                    color = COLOR_WALL
                else:
                    # Normalizuj Q-hodnotu na [0,1] a pouzij cerveno-zelenu colormapu
                    norm = (q_max[r, c] - vmin) / (vmax - vmin)
                    color = plt.cm.RdYlGn(norm)

                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.5, edgecolor=COLOR_GRID, facecolor=color,
                )
                ax.add_patch(rect)

                # Zobraz ciselnu Q-hodnotu v policku
                if val != WALL:
                    ax.text(c + 0.5, r + 0.5, f"{q_max[r, c]:.1f}",
                            ha="center", va="center", fontsize=7)

                # Sipka ukazujuca najlepsiu akciu (politiku agenta)
                if val not in (WALL, TRAP, CHEESE):
                    best = int(np.argmax(agent.q_table[r, c]))
                    ax.annotate(
                        "", xy=(c + 0.5 + ARROW_DX[best] * 0.6, r + 0.5 - ARROW_DY[best] * 0.6),
                        xytext=(c + 0.5, r + 0.5),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    )

        fig.canvas.draw_idle()

    def draw_visits():
        """
        Zobraz pamatovu maticu - pocet navstev kazdeho policka pocas trenovania.

        Tmavsia modra = viac navstev. Ukazuje ktore casti gridu agent
        najviac preskumaval pocas ucenia.
        """
        ax.clear()
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Pamatova matica (pocet navstev)", fontsize=11)

        vmax = max(agent.visit_counts.max(), 1)  # maximum pre normalizaciu

        for r in range(rows):
            for c in range(cols):
                val = env.grid[r, c]
                if val == WALL:
                    color = COLOR_WALL
                else:
                    # Normalizuj pocet navstev na intenzitu modrej farby
                    norm = agent.visit_counts[r, c] / vmax
                    color = plt.cm.Blues(norm * 0.8 + 0.1)

                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.5, edgecolor=COLOR_GRID, facecolor=color,
                )
                ax.add_patch(rect)

                # Zobraz cislo navstev v policku
                count = agent.visit_counts[r, c]
                if count > 0 and val != WALL:
                    ax.text(c + 0.5, r + 0.5, str(count),
                            ha="center", va="center", fontsize=7)

        fig.canvas.draw_idle()

    # ── Klikanie na grid - umiestnenie/odstranenie objektov
    def on_click(event):
        """Handler pre kliknutie na grid - umiestni alebo odstrani vybrany objekt."""
        if event.inaxes != ax:
            return
        c = int(event.xdata)
        r = int(event.ydata)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        # Ak sa klikne na policko s rovnakym typom, odstran ho (toggle)
        current = env.grid[r, c]
        if current == state["tool"]:
            env.grid[r, c] = EMPTY
            if state["tool"] == MOUSE:
                env.mouse_pos = None
        else:
            env.place(r, c, state["tool"])

        # Po zmene gridu treba znova trenovat
        state["trained"] = False
        draw_grid()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── Tlacidla
    btn_h = 0.045
    btn_gap = 0.005
    btn_x = 0.72
    btn_w = 0.25

    def make_btn(y, label, color="#f0f0f0"):
        bax = fig.add_axes([btn_x, y, btn_w, btn_h])
        b = Button(bax, label, color=color, hovercolor="#ddd")
        return b

    y = 0.90
    btn_learn = make_btn(y, "Start learning", "#90EE90")
    y -= btn_h + btn_gap + 0.01
    btn_find = make_btn(y, "Let's find the cheese!", "#98FB98")
    y -= btn_h + btn_gap + 0.02

    btn_mouse = make_btn(y, "Select a mouse", "#FFFACD")
    y -= btn_h + btn_gap
    btn_trap = make_btn(y, "Select a trap", "#FFFACD")
    y -= btn_h + btn_gap
    btn_wall = make_btn(y, "Select a wall", "#FFFACD")
    y -= btn_h + btn_gap
    btn_cheese = make_btn(y, "Select a cheese", "#FFFACD")

    y -= btn_h + btn_gap + 0.02
    btn_show_q = make_btn(y, "Show values of matrix", "#ADD8E6")
    y -= btn_h + btn_gap
    btn_show_mem = make_btn(y, "Show values of memory matrix", "#ADD8E6")

    y -= btn_h + btn_gap + 0.02
    btn_clear = make_btn(y, "Clear grid", "#FFB6B6")

    # Info label
    info_ax = fig.add_axes([btn_x, 0.02, btn_w, 0.08])
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")
    state["info_text"] = info_ax.text(
        0.5, 0.5, "Vyber nastroj a klikni na grid",
        ha="center", va="center", fontsize=9, wrap=True,
    )

    def set_info(text: str):
        state["info_text"].set_text(text)
        fig.canvas.draw_idle()

    # ── Handlere
    def on_mouse(event):
        state["tool"] = MOUSE
        set_info("Klikni na grid pre umiestnenie mysi")

    def on_trap(event):
        state["tool"] = TRAP
        set_info("Klikni na grid pre umiestnenie pasticky")

    def on_wall(event):
        state["tool"] = WALL
        set_info("Klikni na grid pre umiestnenie steny")

    def on_cheese(event):
        state["tool"] = CHEESE
        set_info("Klikni na grid pre umiestnenie syra")

    def on_clear(event):
        env.clear()
        state["trained"] = False
        agent.q_table[:] = 0
        agent.visit_counts[:] = 0
        draw_grid()
        set_info("Grid vymazany")

    def on_learn(event):
        """Spusti trenovanie Q-learning agenta na 1000 epizodach."""
        if env.mouse_pos is None:
            set_info("Najprv umiestni mys!")
            return
        if not env.has_cheese():
            set_info("Najprv umiestni syr!")
            return

        # Trenuj agenta - 1000 epizod, max 200 krokov v kazdej
        history = agent.train(env, n_episodes=1000, max_steps=200)

        # Najdi prvu epizodu kde agent uspesne nasiel syr
        first_success = None
        for i, h in enumerate(history):
            if h["found_cheese"]:
                first_success = i + 1
                break

        # Vypocitaj uspesnost v poslednych 100 epizodach
        last_100 = history[-100:]
        success_rate = sum(1 for h in last_100 if h["found_cheese"]) / len(last_100)

        state["trained"] = True
        draw_grid()

        # Zobraz statistiky trenovania
        msg = f"Natrenované! (1000 epizód)\n"
        if first_success:
            msg += f"Prvy uspech: epizoda {first_success}\n"
        msg += f"Uspesnost (posl. 100): {success_rate:.0%}"
        set_info(msg)

    def on_find(event):
        """Spusti animaciu najlepsej cesty ktoru sa agent naucil."""
        if not state["trained"]:
            set_info("Najprv spusti trenovanie!")
            return

        draw_grid()
        # Ziskaj najlepsiu cestu podla natrenovanej Q-tabulky
        path = agent.get_best_path(env)

        if len(path) <= 1:
            set_info("Agent nenasiel cestu!")
            return

        # Vytvor graficke objekty pre animaciu - bodka (mys) a ciara (stopa)
        dot, = ax.plot([], [], "o", color=COLOR_MOUSE, markersize=18, zorder=10)
        trail_line, = ax.plot([], [], "-", color=COLOR_PATH, linewidth=2, alpha=0.6, zorder=9)

        trail_x = []
        trail_y = []

        def animate(frame):
            """Animacna funkcia - posunie mys o jedno policko v kazdom frame."""
            if frame < len(path):
                r, c = path[frame]
                dot.set_data([c + 0.5], [r + 0.5])
                trail_x.append(c + 0.5)
                trail_y.append(r + 0.5)
                trail_line.set_data(trail_x, trail_y)
            return dot, trail_line

        # Spusti animaciu - 300ms medzi krokmi
        anim = animation.FuncAnimation(
            fig, animate, frames=len(path),
            interval=300, repeat=False, blit=False,
        )
        state["anim"] = anim  # uloz referenciu aby GC nezrusil animaciu

        last_r, last_c = path[-1]
        found = env.grid[last_r, last_c] == CHEESE
        set_info(
            f"Cesta: {len(path)} krokov\n"
            f"{'Syr najdeny!' if found else 'Syr nenajdeny...'}"
        )
        fig.canvas.draw_idle()

    def on_show_q(event):
        if not state["trained"]:
            set_info("Najprv spusti trenovanie!")
            return
        draw_q_values()
        set_info("Q-hodnoty + najlepsia akcia")

    def on_show_mem(event):
        if not state["trained"]:
            set_info("Najprv spusti trenovanie!")
            return
        draw_visits()
        set_info("Pocet navstev pocas trenovania")

    btn_mouse.on_clicked(on_mouse)
    btn_trap.on_clicked(on_trap)
    btn_wall.on_clicked(on_wall)
    btn_cheese.on_clicked(on_cheese)
    btn_clear.on_clicked(on_clear)
    btn_learn.on_clicked(on_learn)
    btn_find.on_clicked(on_find)
    btn_show_q.on_clicked(on_show_q)
    btn_show_mem.on_clicked(on_show_mem)

    draw_grid()
    plt.show()
