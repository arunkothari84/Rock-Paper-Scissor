# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import tkinter as tk
import random
import numpy as np

model = load_model("model.h5")

root = tk.Tk()
root.geometry('100x100')

vid = cv2.VideoCapture(0) 
cv2.namedWindow('image')

def video_capture(result = 'Press "P" to guess!!'):
    while(True):
        ret, frame = vid.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, result, (7, 58),cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (400, 400),
                      (100, 100), (0, 0, 0), 2)
        cv2.imshow('image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('p'): 
            cropped_image = frame[100:400, 100:400]
            cv2.imwrite("capture.jpg",cropped_image)
            result = game()

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

btn = tk.Button(root, text = 'Click to play', bd = '5', command = video_capture)
btn.pack()

def prdiction(img):
    img = load_img(path=img, target_size = (224, 224))
    img = img_to_array(img)
    img = np.array([img])   
    yhat = model.predict(img)
    if (yhat[0][1] > yhat[0][2]) and (yhat[0][1] > yhat[0][0]):
        return "Rock", 0
    elif (yhat[0][0] > yhat[0][1]) and (yhat[0][0] > yhat[0][2]):
        return "Paper", 1
    elif (yhat[0][2] > yhat[0][0]) and (yhat[0][2] > yhat[0][1]):
        return "Scissor", 2

def game():
    choice, index = prdiction('capture.jpg')
    lst = ["Rock", "Paper", "Scissor"]
    comp = random.randint(0, 2)
    if (comp < index or (comp == 2 and index == 0)):
        return f'You did {choice} I did {lst[comp]} "YOU WON!!"'
    elif (comp > index or (comp == 0 and index == 2)):
        return f'You did {choice} I did {lst[comp]} "I WON!!"'
    elif (comp == index):
        return f'You did {choice} I did {lst[comp]} "DRAW!!"'

root.mainloop()
vid.release()
cv2.destroyAllWindows()

