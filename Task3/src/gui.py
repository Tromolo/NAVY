"""
gui.py – Tkinter GUI pre Hopfieldovu siet.

Interaktivna mriezka 10x10 na kreslenie vzorov.
Tlacidla: Save pattern, Repair Sync/Async, Show patterns, Clear grid.
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np

from src.hopfield import HopfieldNetwork

GRID_SIZE = 10
CELL_SIZE = 25


class HopfieldGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hopfield network - patterns")

        self.canvas = tk.Canvas(
            root,
            width=GRID_SIZE * CELL_SIZE,
            height=GRID_SIZE * CELL_SIZE,
            bg="white",
        )
        self.canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

        self.cells: list[list[int]] = []
        self.state = np.full((GRID_SIZE, GRID_SIZE), -1.0, dtype=float)

        for i in range(GRID_SIZE):
            row_ids: list[int] = []
            for j in range(GRID_SIZE):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="black", fill="white",
                )
                row_ids.append(rect_id)
            self.cells.append(row_ids)

        self.canvas.bind("<Button-1>", self.on_click)

        # Tlacidla
        btn_save = tk.Button(root, text="Save pattern", bg="#a8e6a1", command=self.save_pattern)
        btn_save.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        btn_sync = tk.Button(root, text="Repair pattern Sync", bg="#ffeaa7", command=self.repair_sync)
        btn_sync.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        btn_async = tk.Button(root, text="Repair pattern Async", bg="#ffeaa7", command=self.repair_async)
        btn_async.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        btn_show = tk.Button(root, text="Show saved patterns", bg="#74b9ff", command=self.show_patterns_info)
        btn_show.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        btn_clear = tk.Button(root, text="Clear grid", bg="#ff7675", command=self.clear_grid)
        btn_clear.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        tk.Label(
            root,
            text="Max recommended amount\nof saved patterns is 5",
            justify="left",
        ).grid(row=5, column=1, padx=10, pady=5, sticky="sw")

        self.network = HopfieldNetwork(GRID_SIZE * GRID_SIZE)

    def on_click(self, event: tk.Event) -> None:
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            current = self.state[row, col]
            new_val = -1.0 if current > 0 else 1.0
            self.state[row, col] = new_val
            color = "black" if new_val > 0 else "white"
            self.canvas.itemconfig(self.cells[row][col], fill=color)

    def get_flat_state(self) -> np.ndarray:
        return self.state.reshape(-1)

    def set_from_flat_state(self, flat: np.ndarray) -> None:
        self.state = flat.reshape(GRID_SIZE, GRID_SIZE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                val = self.state[i, j]
                color = "black" if val > 0 else "white"
                self.canvas.itemconfig(self.cells[i][j], fill=color)

    def save_pattern(self) -> None:
        pattern = self.get_flat_state()
        self.network.add_pattern(pattern)
        messagebox.showinfo(
            "Pattern saved",
            f"Pattern stored. Total patterns: {len(self.network.stored_patterns)}",
        )

    def repair_sync(self) -> None:
        if not self.network.stored_patterns:
            messagebox.showwarning("No patterns", "Save at least one pattern first.")
            return
        noisy = self.get_flat_state()
        recovered = self.network.recover_sync(noisy)
        self.set_from_flat_state(recovered)

    def repair_async(self) -> None:
        if not self.network.stored_patterns:
            messagebox.showwarning("No patterns", "Save at least one pattern first.")
            return
        noisy = self.get_flat_state()
        recovered = self.network.recover_async(noisy)
        self.set_from_flat_state(recovered)

    def show_patterns_info(self) -> None:
        count = len(self.network.stored_patterns)
        if count == 0:
            msg = "No patterns stored yet."
        else:
            msg = f"Stored patterns: {count}\n\nYou can draw a noisy version\nand use Sync/Async repair."
        messagebox.showinfo("Stored patterns", msg)

    def clear_grid(self) -> None:
        self.state.fill(-1.0)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.itemconfig(self.cells[i][j], fill="white")
