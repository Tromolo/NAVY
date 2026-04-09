import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from .lsystem import PRESETS, apply_rules, interpret


def run_gui() -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.68, bottom=0.05, top=0.95)

    fig.suptitle("L-systems drawing", fontsize=14, fontweight="bold")
    state = {
        "start_x": 200,
        "start_y": 200,
        "start_angle_deg": 0.0,
        "start_angle_rad": None,
        "nesting": 3,
        "line_size": 5,
    }

    def draw_lsystem(axiom: str, rule: str, angle_deg: float):
        ax.clear()

        start_angle = state["start_angle_deg"]
        if state["start_angle_rad"] is not None:
            start_angle = math.degrees(state["start_angle_rad"])

        # Aplikuj pravidla prepisovania (nesting)
        instructions = apply_rules(axiom, rule, state["nesting"])

        # Interpretuj vysledny retazec ako korytnaci grafiku
        segments = interpret(
            instructions,
            step_length=state["line_size"],
            angle_deg=angle_deg,
            start_x=state["start_x"],
            start_y=state["start_y"],
            start_angle_deg=start_angle,
        )

        # Vykresli vsetky usecky
        for x1, y1, x2, y2, draw in segments:
            if draw:
                ax.plot([x1, x2], [y1, y2], "k-", linewidth=0.8)

        ax.set_aspect("equal")
        ax.set_title(
            f"Axiom: {axiom} | Rule: F → {rule} | "
            f"Angle: {angle_deg}° | Nesting: {state['nesting']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw_idle()

    def clear_canvas():
        """Vymaz platno."""
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw_idle()

    # ── Tlacidla a vstupne polia (vpravo od platna)
    btn_h = 0.035
    btn_gap = 0.005
    btn_x = 0.72
    btn_w = 0.25
    lbl_h = 0.018
    input_h = 0.03
    input_w = 0.15

    y = 0.94  # zaciatok zhora

    # ── Vstupne polia pre nastavenie parametrov

    def make_label(y_pos, text):
        """Vytvor textovy label nad vstupnym polom."""
        lax = fig.add_axes([btn_x, y_pos, btn_w, lbl_h])
        lax.axis("off")
        lax.text(0, 0.5, text, fontsize=8, va="center")
        return lax

    def make_input(y_pos, initial_text=""):
        """Vytvor vstupne textove pole."""
        iax = fig.add_axes([btn_x, y_pos, input_w, input_h])
        tb = TextBox(iax, "", initial=initial_text)
        return tb

    def make_btn(y_pos, label, color="#f0f0f0"):
        """Vytvor tlacidlo."""
        bax = fig.add_axes([btn_x, y_pos, btn_w, btn_h])
        b = Button(bax, label, color=color, hovercolor="#ddd")
        return b

    # Pociatocna X pozicia
    make_label(y, "Starting X position (int)")
    y -= lbl_h + btn_gap
    tb_start_x = make_input(y, str(state["start_x"]))
    y -= input_h + btn_gap + 0.005

    # Pociatocna Y pozicia
    make_label(y, "Starting Y position (int)")
    y -= lbl_h + btn_gap
    tb_start_y = make_input(y, str(state["start_y"]))
    y -= input_h + btn_gap + 0.005

    # Pociatocny uhol v stupnoch
    make_label(y, "Starting angle (degree)")
    y -= lbl_h + btn_gap
    tb_start_angle_deg = make_input(y, "")
    y -= input_h + btn_gap + 0.005

    # Pociatocny uhol v radianoch
    make_label(y, "Starting angle (radians)")
    y -= lbl_h + btn_gap
    tb_start_angle_rad = make_input(y, "")
    y -= input_h + btn_gap + 0.005

    # Pocet vnoreni (nesting)
    make_label(y, "The number of nesting (int)")
    y -= lbl_h + btn_gap
    tb_nesting = make_input(y, str(state["nesting"]))
    y -= input_h + btn_gap + 0.005

    # Velkost ciary
    make_label(y, "Size of the line (int)")
    y -= lbl_h + btn_gap
    tb_line_size = make_input(y, str(state["line_size"]))
    y -= input_h + btn_gap + 0.015

    # ── Tlacidla pre preddefinovane L-systemy
    btn_draw1 = make_btn(y, "Draw first", "#90EE90")
    y -= btn_h + btn_gap
    btn_draw2 = make_btn(y, "Draw second", "#90EE90")
    y -= btn_h + btn_gap
    btn_draw3 = make_btn(y, "Draw third", "#90EE90")
    y -= btn_h + btn_gap
    btn_draw4 = make_btn(y, "Draw fourth", "#90EE90")
    y -= btn_h + btn_gap + 0.015

    # ── Sekcia pre vlastny L-system
    lax = fig.add_axes([btn_x, y, btn_w, lbl_h + 0.005])
    lax.axis("off")
    lax.text(0, 0.5, "Custom", fontsize=10, fontweight="bold", va="center")
    y -= lbl_h + btn_gap + 0.005

    # Vlastny axiom
    make_label(y, "Axiom (F,+,-,[,])")
    y -= lbl_h + btn_gap
    tb_axiom = make_input(y, "")
    y -= input_h + btn_gap + 0.005

    # Vlastne pravidlo
    make_label(y, "Rule (F,+,-,[,])")
    y -= lbl_h + btn_gap
    tb_rule = make_input(y, "")
    y -= input_h + btn_gap + 0.005

    # Vlastny uhol v stupnoch
    make_label(y, "Angle (degree)")
    y -= lbl_h + btn_gap
    tb_angle_deg = make_input(y, "")
    y -= input_h + btn_gap + 0.005

    # Vlastny uhol v radianoch
    make_label(y, "Angle (radians)")
    y -= lbl_h + btn_gap
    tb_angle_rad = make_input(y, "")
    y -= input_h + btn_gap + 0.015

    # Tlacidlo pre kreslenie vlastneho L-systemu
    btn_draw_custom = make_btn(y, "Draw custom", "#FFB347")
    y -= btn_h + btn_gap + 0.005

    # Tlacidlo pre vymazanie platna
    btn_clear = make_btn(y, "Clear canvas", "#FFB6B6")

    # ── Funkcie pre aktualizaciu stavu z vstupnych poli

    def update_state():
        """Nacitaj hodnoty zo vsetkych vstupnych poli do stavu aplikacie."""
        try:
            val = tb_start_x.text.strip()
            if val:
                state["start_x"] = float(val)
        except ValueError:
            pass

        try:
            val = tb_start_y.text.strip()
            if val:
                state["start_y"] = float(val)
        except ValueError:
            pass

        try:
            val = tb_start_angle_deg.text.strip()
            if val:
                state["start_angle_deg"] = float(val)
                state["start_angle_rad"] = None
            else:
                state["start_angle_deg"] = 0.0
        except ValueError:
            pass

        try:
            val = tb_start_angle_rad.text.strip()
            if val:
                state["start_angle_rad"] = float(val)
        except ValueError:
            pass

        try:
            val = tb_nesting.text.strip()
            if val:
                state["nesting"] = int(val)
        except ValueError:
            pass

        try:
            val = tb_line_size.text.strip()
            if val:
                state["line_size"] = float(val)
        except ValueError:
            pass

    # ── Handlery pre tlacidla

    def on_draw_preset(preset_id):
        """Vykresli preddefinovany L-system podla jeho ID."""
        def handler(event):
            update_state()
            p = PRESETS[preset_id]
            draw_lsystem(p["axiom"], p["rule"], p["angle"])
        return handler

    def on_draw_custom(event):
        """Vykresli vlastny L-system zo vstupnych poli."""
        update_state()

        axiom = tb_axiom.text.strip()
        rule = tb_rule.text.strip()

        if not axiom or not rule:
            return

        # Uhol - stupne maju prioritu, inak pouzi radiany
        angle_deg = 0.0
        try:
            val = tb_angle_deg.text.strip()
            if val:
                angle_deg = float(val)
            else:
                val = tb_angle_rad.text.strip()
                if val:
                    angle_deg = math.degrees(float(val))
        except ValueError:
            pass

        draw_lsystem(axiom, rule, angle_deg)

    def on_clear(event):
        """Vymaz platno."""
        clear_canvas()

    # ── Pripojenie handlerov k tlacidlam
    btn_draw1.on_clicked(on_draw_preset(1))
    btn_draw2.on_clicked(on_draw_preset(2))
    btn_draw3.on_clicked(on_draw_preset(3))
    btn_draw4.on_clicked(on_draw_preset(4))
    btn_draw_custom.on_clicked(on_draw_custom)
    btn_clear.on_clicked(on_clear)

    # Inicializuj prazdne platno
    clear_canvas()
    plt.show()
