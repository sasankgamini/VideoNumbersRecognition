import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import cv2
from keras.models import load_model

model = load_model('savedDigits.h5')

capture=cv2.VideoCapture(0) #starts the camera
while True:
    status, frame = capture.read()
    grayimage=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredgray=cv2.GaussianBlur(grayimage,(5,5),0)
    ret, im_th = cv2.threshold(blurredgray, 110, 255, cv2.THRESH_BINARY_INV)

    ctrs, hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctrs, -1,(255,0,0),3)
    
    rectangles=[]
    for eachContour in ctrs:
        rectangles.append(cv2.boundingRect(eachContour))
    for eachRectangle in rectangles:
        ROI = im_th[eachRectangle[1]-40:eachRectangle[1]+eachRectangle[3]+40,eachRectangle[0]-40:eachRectangle[0]+eachRectangle[2]+40]
        if ROI.any():
            imgarray=cv2.resize(ROI,(28,28))
            dilatedimg=cv2.dilate(imgarray,(3,3)) #this is to thicken
            dilatedlist=[dilatedimg]
            dilatedarray=np.array(dilatedlist)
            dilatedarray=dilatedarray/255
            predictions=model.predict(dilatedarray)
##            print(predictions[0])
##            print(np.argmax(predictions[0]))
            cv2.putText(frame, str(np.argmax(predictions[0])), (eachRectangle[0]-10, eachRectangle[1]-50),cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.rectangle(frame,(eachRectangle[0]-10,eachRectangle[1]-10),(eachRectangle[0]+eachRectangle[2]+10,eachRectangle[1]+eachRectangle[3]+10),(0,0,255),2)

    cv2.imshow('image',frame)    
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

