from src.gui import HopfieldGUI
import tkinter as tk


def main() -> None:
    root = tk.Tk()
    HopfieldGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
