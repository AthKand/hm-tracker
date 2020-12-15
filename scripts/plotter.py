# -*- coding: utf-8 -*-
# flags for after active and inactive
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog

import numpy as np
import cv2

def fopen():
    root.filename = filedialog.askopenfilename(title = "Select file", filetypes = (("All Files", "*.*"), ("Log Files", "*.log")))
    return root.filename 
    
def log_reader(fname):
    flag = False
    num = None
    savelist = []
    count = 0

    name = str(fname)

    with open(name) as file:
        frame =  np.zeros(shape=[712 , 1176, 3], dtype = np.uint8)
        lines = file.readlines()
        points = []
        count = 0 
        
        for line in lines:
            if 'Recording ' in line:
                flag = True
                count += 1
            
            if flag:
                if "(" in line:
                    coords = line.partition("(")[2].partition(")")[0].split(',')
                    points.append((int(coords[0]), int(coords[1])))
                    count += 1 


        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.circle(frame, points[i], 1, color = (255, 255, 255), thickness = -1)

        cv2.imshow('trial path', frame)
             
if __name__ == "__main__":
    
    root = Tk()
    root.title('Converter')
    file = fopen()
    log_path = Path(file).resolve()

    log_reader(log_path) 
    my_label = Label(root, text = 'Your file has been converted successfully').pack()
    root.mainloop()
           