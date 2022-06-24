import tkinter as tk
from tkinter import ttk

class ScaleSlider(ttk.Scale):
    def __init__(self, master, master_style, *args, **kwargs):
        kwargs.setdefault("orient", "horizontal")
        self.variable = kwargs.pop('variable', tk.IntVar(master))
        ttk.Scale.__init__(self, master, variable=self.variable, **kwargs)
        self._style_name = '{}.custom.{}.TScale'.format(self, kwargs['orient'].capitalize()) # unique style name to handle the text
        self['style'] = self._style_name
        self.variable.trace_add('write', self._update_text)
        self.master_style = master_style
        self._update_text()

    def _update_text(self, *args):
        self.master_style.configure(self._style_name, text="{:.1f}".format(self.variable.get()))
