from manimlib import *
from geonerative.shell import log_name, read_code

manim_objects = {}


class Script(Scene):
    def construct(self):
        log_ = open(log_name, "r")
        text = log_.read()
        print(text)
        read_code(self, text, log_name)
