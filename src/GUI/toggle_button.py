import tkinter as tk
class ToggleButton(tk.Button):

    def __init__(self, master, function, isOn=True, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.on  = tk.PhotoImage(file='../misc/on.png')
        self.off = tk.PhotoImage(file='../misc/off.png')

        self.value = isOn
        self.function = function
        self.config(bd=0, command=self.onClick)
        if self.value:
            self.config(image=self.on)
        else:
            self.config(image=self.off)

    def onClick(self):                            
        if self.value:
            self.config(image = self.off)
            self.value = False
        else:
            self.config(image = self.on)
            self.value = True

        self.function()

