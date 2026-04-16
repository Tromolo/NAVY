import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from .fractal import mandelbrot, julia


# Vytvor custom colormapu s HSV farbami (hue + saturacia)
def create_fractal_cmap():
    colors = [
        (0.0, 0.0, 0.0),       # cierna (body v mnozine)
        (0.0, 0.4, 0.0),       # tmavo zelena
        (0.0, 1.0, 0.0),       # zelena
        (0.5, 1.0, 0.0),       # zlto-zelena
        (1.0, 1.0, 0.0),       # zlta
        (1.0, 0.6, 0.0),       # oranzova
        (1.0, 0.2, 0.0),       # cerveno-oranzova
        (0.8, 0.1, 0.0),       # tmavo cervena
    ]
    return mcolors.LinearSegmentedColormap.from_list("fractal", colors, N=256)


class FractalViewer:
    def __init__(self, fractal_type: str = "mandelbrot",
                 julia_c: complex = -0.7 + 0.27015j,
                 width: int = 800, height: int = 600, max_iter: int = 256):
        self.fractal_type = fractal_type
        self.julia_c = julia_c
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.cmap = create_fractal_cmap()

        # Povodne hranice
        if fractal_type == "mandelbrot":
            self.default_bounds = (-2.0, 1.0, -1.0, 1.0)
        else:
            self.default_bounds = (-1.5, 1.5, -1.5, 1.5)

        self.xmin, self.xmax, self.ymin, self.ymax = self.default_bounds

    def compute(self) -> np.ndarray:
        """Vypocitaj fraktal pre aktualne hranice."""
        if self.fractal_type == "mandelbrot":
            return mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax,
                              self.width, self.height, self.max_iter)
        else:
            return julia(self.xmin, self.xmax, self.ymin, self.ymax,
                         self.width, self.height, self.julia_c, self.max_iter)

    def render(self) -> None:
        """Vykresli fraktal a nastav zoom handler."""
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

        iterations = self.compute()
        self.img = self.ax.imshow(
            iterations,
            extent=(self.xmin, self.xmax, self.ymin, self.ymax),
            cmap=self.cmap,
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
        )

        title = "Mandelbrot set" if self.fractal_type == "mandelbrot" else f"Julia set (c = {self.julia_c})"
        self.ax.set_title(f"{title}\nLavy klik = zoom in | Pravy klik = zoom out | Stredny = reset")
        self.ax.set_xlabel("Re")
        self.ax.set_ylabel("Im")

        # Pripoj event handler pre zoom
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event) -> None:
        """Handler pre kliknutie mysi — zoom in/out/reset."""
        if event.inaxes != self.ax:
            return

        cx, cy = event.xdata, event.ydata
        dx = (self.xmax - self.xmin)
        dy = (self.ymax - self.ymin)

        if event.button == 1:
            # Lavy klik — zoom in 2x
            self.xmin = cx - dx / 4
            self.xmax = cx + dx / 4
            self.ymin = cy - dy / 4
            self.ymax = cy + dy / 4
        elif event.button == 3:
            # Pravy klik — zoom out 2x
            self.xmin = cx - dx
            self.xmax = cx + dx
            self.ymin = cy - dy
            self.ymax = cy + dy
        elif event.button == 2:
            # Stredne tlacidlo — reset
            self.xmin, self.xmax, self.ymin, self.ymax = self.default_bounds

        self._update()

    def _update(self) -> None:
        """Prepocitaj a aktualizuj obrazok."""
        iterations = self.compute()
        self.img.set_data(iterations)
        self.img.set_extent((self.xmin, self.xmax, self.ymin, self.ymax))
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.fig.canvas.draw_idle()


def run_gui(max_iter: int = 256) -> None:
    """Spusti vizualizaciu Mandelbrotovej a Juliovej mnoziny."""
    # Mandelbrot
    viewer1 = FractalViewer("mandelbrot", max_iter=max_iter)
    viewer1.render()

    # Julia
    viewer2 = FractalViewer("julia", julia_c=-0.7 + 0.27015j, max_iter=max_iter)
    viewer2.render()

    plt.show()
