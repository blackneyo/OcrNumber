from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
model = load_model('mnist.h5')

def predict_digit(img):
    #이미지 28x28 픽셀로 크기 조정
    img = img.resize((28,28))
    #rgb를 회색조로 변환
    img = img.convert('L')
    img = np.array(img)
    #모델 정규화
    img = img.reshape(1,28,28,1)
    img = img/255
    #클래스 예측 res에 대한 최대값 res는 모델에 있는 이미지에 대한 에측
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
'''
2022.04.06 이동한
초기화 함수 정의 
self는 tk.TK 즉 App
x와 y를 0으로 초기화
tk.Canvas 폭 300, 높이 300, 백그라운드 흑색 
사각형 틀을 구성한 후 안에 글씨를 백색원으로 드래그해서 그릴 수 있게 설정
정확도 표시
'''
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "black", cursor="cross")
        self.label = tk.Label(self, text="Analyzing..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "인식버튼", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "지우기", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HD = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HD)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='white')

app = App()
mainloop()