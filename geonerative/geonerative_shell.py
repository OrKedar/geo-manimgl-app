import geonerative.Geometric_Category as GC
from geonerative.language import run as run_language
from geonerative.language import global_objects
from manimlib import *
import tkinter as tk
from tkinter import ttk
from random import randint
from functools import partial

manim_objects = {}


class Script(Scene):
    def construct(self):
        def read_code(code):
            print(code.get("1.0", "end")[:-1])
            result, error = run_language('<stdin>', code.get("1.0", "end")[:-1])

            if error:
                print(error.as_string())
            elif result:
                for id_, obj in global_objects.items():
                    if obj.type == 'circle':
                        arc_center = np.array([obj.x_loc, obj.y_loc, 0])
                        manim_objects[id_] = Circle(radius=obj.radius, arc_center=arc_center, fill_color=obj.color,
                                                    fill_opacity=1.0, stroke_color=obj.color, stroke_opacity=1.0)
                        self.play(manim_objects[id_].animate.shift(RIGHT * 0), run_time=0.1)

        root = tk.Tk()
        root.grid()
        code_Box = tk.Text(root, height=10, width=90)
        code_Box.grid(column=0, row=0)
        tk.ttk.Button(root, text="Run", command=partial(read_code, code_Box)).grid(column=0, row=1)

        root.mainloop()
