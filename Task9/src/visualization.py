import tkinter as tk
from tkinter import ttk, colorchooser, messagebox

from .fractal import midpoint_displacement


CANVAS_WIDTH = 901
CANVAS_HEIGHT = 600


class FractalTerrainGUI:
    """Tkinter GUI - generuje fraktalny terain pomocou midpoint displacement.

    Pouzivatel zada start/end suradnice, pocet iteracii, offset a farbu,
    klikne Draw - terain sa prikresli na platno. Viackrat = vrstvenie.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FractalTerrain2D by Tomas Hruby")
        self.root.resizable(False, False)

        self.selected_color = "#008000"  # default zelena
        self._build_ui()

    def _build_ui(self) -> None:
        # platno - lava strana
        self.canvas = tk.Canvas(
            self.root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg="white", highlightthickness=1, highlightbackground="black"
        )
        self.canvas.grid(row=0, column=0, padx=8, pady=8, rowspan=20)

        # panel s ovladanim - prava strana
        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        self.entries: dict = {}

        fields = [
            ("start_x", "Start X possition (float)", "0"),
            ("start_y", "Start Y possition (float)", "450.5"),
            ("end_x",   "End X possition (float)",   "901"),
            ("end_y",   "End Y possition (float)",   "450.5"),
            ("iters",   "The number of iteration (int)", "8"),
            ("offset",  "Offset size (float)",       "30"),
        ]
        for i, (key, label, default) in enumerate(fields):
            ttk.Label(panel, text=label).grid(row=i * 2, column=0, sticky="w", pady=(6, 0))
            e = ttk.Entry(panel, width=15)
            e.insert(0, default)
            e.grid(row=i * 2 + 1, column=0, sticky="w")
            self.entries[key] = e

        # zobrazenie aktualnej farby
        self.color_label = ttk.Label(panel, text=f"Selected color is {self.selected_color}")
        self.color_label.grid(row=12, column=0, sticky="w", pady=(10, 2))

        self.color_swatch = tk.Label(panel, bg=self.selected_color, width=20, height=2,
                                     relief="ridge", borderwidth=1)
        self.color_swatch.grid(row=13, column=0, sticky="w", pady=(0, 6))

        tk.Button(panel, text="Pick a color", bg="#fff7b0",
                  command=self._pick_color).grid(row=14, column=0, sticky="w", pady=2)

        tk.Button(panel, text="Draw", bg="#c8f7c5",
                  command=self._draw).grid(row=15, column=0, sticky="w", pady=2)

        tk.Button(panel, text="Clear canvas", bg="#f7c5c5",
                  command=self._clear).grid(row=16, column=0, sticky="w", pady=2)

    def _pick_color(self) -> None:
        result = colorchooser.askcolor(initialcolor=self.selected_color, title="Vyber farbu")
        if result and result[1]:
            self.selected_color = result[1]
            self.color_label.config(text=f"Selected color is {self.selected_color}")
            self.color_swatch.config(bg=self.selected_color)

    def _read_params(self):
        try:
            sx = float(self.entries["start_x"].get())
            sy = float(self.entries["start_y"].get())
            ex = float(self.entries["end_x"].get())
            ey = float(self.entries["end_y"].get())
            it = int(self.entries["iters"].get())
            off = float(self.entries["offset"].get())
        except ValueError:
            messagebox.showerror("Chyba", "Skontroluj vstupne hodnoty - musia byt cisla.")
            return None
        if it < 0 or it > 14:
            messagebox.showerror("Chyba", "Pocet iteracii musi byt 0-14 (vyssie = velmi pomale).")
            return None
        return sx, sy, ex, ey, it, off

    def _draw(self) -> None:
        params = self._read_params()
        if params is None:
            return
        sx, sy, ex, ey, iters, offset = params

        # vygeneruj body terainu
        points = midpoint_displacement((sx, sy), (ex, ey), iters, offset)

        # uzavri polygon - body terainu + pravy dolny + lavy dolny roh
        flat = []
        for x, y in points:
            flat.extend([x, y])
        # roh pri pravom konci dole
        flat.extend([points[-1][0], CANVAS_HEIGHT])
        # roh pri lavom zaciatku dole
        flat.extend([points[0][0], CANVAS_HEIGHT])

        self.canvas.create_polygon(flat, fill=self.selected_color, outline=self.selected_color)

    def _clear(self) -> None:
        self.canvas.delete("all")


def run_gui() -> None:
    root = tk.Tk()
    FractalTerrainGUI(root)
    root.mainloop()
