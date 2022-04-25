#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:52:06 2022

@author: ubuntu-usn
"""

import cv2

from models import model
from load_data import classify

word_list ={
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y'}

x=0.6  # start point/total width
y=0.6  

checkpoint_path = 'checkpoint/cp.ckpt'

model = model()
model.load_weights(checkpoint_path)

def launch_webcam():
    cap = cv2.VideoCapture(0)
    word = 'Y'

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        # cv2.rectangle(frame, (int(x * frame.shape[1]), 1),
        #              (frame.shape[1], int(y * frame.shape[0])), (255, 110, 180), 2) #drawing ROI
        
        cv2.rectangle(frame, (440,2),
                     (640, 250), (255, 110, 180), 2) 
    
        # Display the resulting frame
        img = frame
        frame=cv2.putText(frame,word,(445,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            
            img = img[0:int(y * frame.shape[0]),
                        int(x * frame.shape[1]):frame.shape[1]]  # clip the ROI
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("images/asl.jpg",gray)
            pred=classify(model)            
            word = word_list[pred[-1]]
            
            # cv2.imshow('gray', gray)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
            
        if cv2.waitKey(1) & 0xFF == ord('q'):           

            break
        
        
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    launch_webcam()