import geonerative.Geometric_Category as GC
from geonerative.language import run as run_language
from geonerative.language import global_objects
from manimlib import *
import tkinter as tk
from tkinter import ttk
from random import randint
from functools import partial
from geonerative.token_names import *

manim_objects = {}
log_name = "geonerative\\code_log.txt"
script_name = "Script"


def read_copy_code(scene, text, log_name_):
    if read_code(scene, text, log_name_):
        log_ = open(log_name_, "a")
        log_.write(text)
        log_.close()
    return


def read_code(scene, text, log_name_):
    for id_, obj in global_objects.items():
        scene.remove(manim_objects[id_])
    result, error = run_language('<stdin>', text)

    if error:
        print(error.as_string())
    elif result:
        draw_objects(scene, global_objects)

    return error is None


def draw_objects(scene, objects):
    for id_, obj in global_objects.items():
        properties=obj.properties
        if obj.type == 'circle':
            arc_center = np.array([properties['x_loc'], properties['y_loc'], 0])
            manim_objects[id_] = Circle(radius=properties['radius'], arc_center=arc_center, fill_color=properties['color'],
                                        fill_opacity=1.0, stroke_color=properties['color'], stroke_opacity=1.0)
            scene.play(manim_objects[id_].animate.shift(RIGHT * 0), run_time=0.01)


def read_copy_code_from_tk(scene, text_box, log_name_):
    return read_copy_code(scene, text_box.get("1.0", "end")[:-1], log_name_)


class Script(Scene):
    def construct(self):
        code_log = open(log_name, "w")
        code_log.close()
        root = tk.Tk()
        root.grid()
        code_box = tk.Text(root, height=10, width=90)
        code_box.grid(column=0, row=0)
        cmd=partial(read_copy_code_from_tk, self, code_box, log_name)
        tk.ttk.Button(root, text="Run", command=cmd).grid(column=0, row=1)

        root.mainloop()
