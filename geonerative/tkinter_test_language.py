import tkinter as tk
from tkinter import ttk
from random import randint
from functools import partial
import Geometric_Category as GC
from language import run as language_run
from language import global_objects

WIDTH, HEIGHT = 200, 200

if __name__ == "__main__":
    root = tk.Tk()
    root.grid()
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, borderwidth=0, highlightthickness=0,
                       bg="black")
    canvas.grid(column=0, row=0)
    artBox = tk.ttk.Frame(root, padding=10)
    artBox.grid(column=1, row=0)
    code_string = tk.StringVar()
    code_Box = tk.Text(artBox, height=10, width=50)
    code_Box.pack()
    code_Box.grid(column=0, row=2)


    def _create_circle(self, x, y, r, **kwargs):
        return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)


    def draw(canvas_):
        canvas_.delete("all")
        for id_, obj in global_objects.items():
            obj.draw(canvas_)


    def read_code(text, canvas_):
        text = text.get('1.0', 'end')[:-1]
        result, error = language_run('whatever', text)
        if error:
            print(error.as_string())
        else:
            print(text)
            print("result: ",type(result),': ',result)
            print("Objects:\n--------")
        for key, item in global_objects.items():
            print(f"id {key} ---- {item}")
        draw(canvas_)


    # properties = {'x_loc': 100, 'y_loc': 100, 'radius': 50, 'color': '#BBB'}
    # global_objects['1'] = GC.Geo('circle', '1', properties)

    tk.Canvas.create_circle = _create_circle
    draw(canvas)
    create_Circle = tk.ttk.Button(artBox, text="Run",
                                  command=partial(read_code, code_Box, canvas)).grid(column=0, row=1)

    root.mainloop()
