import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
from keras.models import load_model


class DrawDigitTk:
    def __init__(self, path):
        self.model = load_model(path)

        self.root = tk.Tk()
        self.RES = self.W, self.H = 210, 250
        self.root.geometry(f'{self.W}x{self.H}')
        self.root.resizable(False, False)

        self.color_bg = 'black'
        self.color_draw = 'white'

        self.w = self.h = 200
        self.size_pen = 7
        self.frame = tk.Canvas(width=self.w, height=self.h, bg=self.color_bg)
        self.frame.bind('<B1-Motion>', self.draw)
        self.frame.place(x=0, y=40)

        self.button = tk.Button(self.root, text='get_array')
        self.button.bind('<Button-1>', self.click_b1)
        self.button.place(x=10, y=10)

        self.b_clear = tk.Button(self.root, text='clear')
        self.b_clear.bind('<Button-1>', self.click_b2)
        self.b_clear.place(x=80, y=10)

        self.label = tk.Label(text='Цифра')
        self.label.place(x=150, y=10)

    def draw(self, event: tk.Event):
        size = self.size_pen
        x1 = event.x - size
        x2 = event.x + size
        y1 = event.y - size
        y2 = event.y + size

        self.frame.create_oval(x1, y1, x2, y2, fill=self.color_draw, outline=self.color_draw)

    @staticmethod
    def get_image_screenshot(obj: tk.Canvas):
        x0, y0 = obj.winfo_rootx(), obj.winfo_rooty()
        x1, y1 = x0 + obj.winfo_width(), y0 + obj.winfo_height()

        return ImageGrab.grab(bbox=(x0+2, y0+2, x1-2, y1-2)).convert('L')

    def get_digit_2828(self) -> np.array:

        scr = self.get_image_screenshot(self.frame)
        img = np.array(scr)

        array = np.zeros((28, 28))

        cell = 200//28
        for i, y in enumerate(range(0, self.h, cell)):
            for j, x in enumerate(range(0, self.w, cell)):
                self.frame.create_rectangle(y, x, y + cell, x + cell, fill=self.color_bg)

                rect = img[x:x+cell, y:y+cell]
                sum = np.sum(rect)/(255*cell**2) * 100 * 2.5
                c = int(sum)

                if c > 185:
                    rgb = (c, c, c)
                    color = "#%02x%02x%02x" % rgb

                    self.frame.create_rectangle(y, x, y+cell, x+cell, fill=color)

                    if sum:
                        array[j, i] = c

        return array

    def click_b1(self, event):
        img_array = self.get_digit_2828()/255

        model = self.model
        x = np.expand_dims(img_array, axis=0)
        res = model.predict(x)
        digit = np.argmax(res)
        self.label.configure(text=str(digit))
        print(digit)

    def click_b2(self, event):
        self.frame.delete('all')

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    path_model = r'find_digit.h5'
    # path_model = r'E:\Piton\Нейронные сети\Литература\keras\5. СNN\find_digit_cnn.h5'

    app = DrawDigitTk(path_model)
    app.run()