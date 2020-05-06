import cv2
import numpy as np

#Create a body classifier
ped_classifier=cv2.CascadeClassifier('C:\\Users\Hit\Master OpenCV\Haarcascades\haarcascade_fullbody.xml')

#Initiate viceo capture
cap=cv2.VideoCapture('C:\\Users\Hit\Master OpenCV\images\walking.avi')

while cap.isOpened():
        
    ret,frame=cap.read()
    frame=cv2.resize(frame,None,fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #feed the frame to classifier
    peds=ped_classifier.detectMultiScale(gray,1.2,3)
    
    for(x,y,w,h) in peds:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('Pedestrians',frame)
    if(cv2.waitKey(1)==13):
        break

cap.release()
cv2.destroyAllWindows()