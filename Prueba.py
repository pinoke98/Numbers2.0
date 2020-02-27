import numpy as np
import cv2
from sklearn.externals import joblib

contMLP=0
contKNN=0
cap = cv2.VideoCapture(0)

while(1):
    ret,frame = cap.read()
    frame_resize = cv2.resize(frame,(100,100), interpolation = cv2.INTER_AREA)
    cv2.imshow("img",frame_resize)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,7)
    ret,edges = cv2.threshold(blur,0,300,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #edges = cv2.adaptiveThreshold(blur,230,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges= cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("edges",edges)
    #cv2.waitKey(0)
    
    drawing = np.zeros(frame.shape,np.uint8)
    
    contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    draw = cv2.drawContours(frame,contours,-1,(0,0,255),3)
    #cv2.imshow("contours",draw)
    #cv2.waitKey(0)
    for component in zip(contours,hierarchy):
        currentContour = component[0]
        x,y,w,h = cv2.boundingRect(currentContour)
        #print(x,y,w,h)
        p = cv2.arcLength(currentContour,True)   
        epsilon = 0.015*p
        approx = cv2.approxPolyDP(currentContour,epsilon,True)
        lados = len(approx)
        #print(lados)
        #cv2.imshow("lados", draw)
        #cv2.waitKey(0)
        
        empty = np.zeros((h,w),np.uint8)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        #print(w,h)
        roi= edges[y:y+h,x:x+w]
        empty[0:h,0:w]=roi
        #cv2.imshow("roi",roi)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        edges = empty
        #cv2.imshow("empty", edges)
        #cv2.waitKey(0)
        img_resize = cv2.resize(edges,(100,100), interpolation = cv2.INTER_AREA)
        ret, edges_res = cv2.threshold(img_resize,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow("img_resize",img_resize)
        #cv2.waitKey(0)
        
        contours_2,hierarchy_2 = cv2.findContours(edges_res.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hier = hierarchy[0]
        contornos=hier.shape[0]
cv2.destroyAllWindows()

