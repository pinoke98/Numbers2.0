import numpy as np
import cv2
from glob import glob
from sklearn.externals import joblib

contMLP=0
contKNN=0

for j in range(3):
    img_Carpeta = ('Pruebas/'+str(j)+'/'+str(j)+'   (*.jpg')
    img_names = glob(img_Carpeta)
    #print(img_names)

    for fn in img_names:
        img = cv2.imread(fn,1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,7)
        ret,edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edges= cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("edges",edges)
        #cv2.waitKey(0)
        
        drawing = np.zeros(img.shape,np.uint8)
        
        contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        draw = cv2.drawContours(img,contours,-1,(0,0,255),3)
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
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            #print(w,h)
            roi= edges[y:y+h,x:x+w]
            empty[0:h,0:w]=roi
            #cv2.imshow("roi",roi)
            #cv2.imshow("edges", edges)
            #cv2.waitKey(0)
            edges = empty
            #cv2.imshow("empty", edges)
            #cv2.waitKey(0)
            img_resize = cv2.resize(edges,(100,100), interpolation = cv2.INTER_AREA)
            ret, edges_res = cv2.threshold(img_resize,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow("img_resize",img_resize)
            cv2.waitKey(0)
            
            contours_2,hierarchy_2 = cv2.findContours(edges_res.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            hier = hierarchy[0]
            contornos=hier.shape[0]
            
            for component_2 in zip(contours_2, hierarchy_2):
                currentContour_2 =  component_2[0]
                ncont=len(currentContour_2)
                x,y,w,h = cv2.boundingRect(currentContour_2)
                M = cv2.moments(currentContour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                A = cv2.contourArea(currentContour_2)
                Rect=A/w*h
                p = cv2.arcLength(currentContour_2,True)
                RA = w/float(h)
                #print(cx,cy,A,p,RA)
                h,w = edges_res.shape
                Hu = cv2.HuMoments(M)
                aS=0
                aI=0
                aR=0
                aL=0
                aT=0
                aCH=0
                aCV=0
                
                for f in range(int(h/3),int((2*h)/3)):
                    for c in range(int(w)):
                        if(edges_res[f,c]==255):
                            aCH+=1
                for f in range(h):
                    for c in range(int(w/3),int((2*w)/3)):
                        if(edges_res[f,c]==255):
                            aCV+=1
                for f in range(int(h/2)):
                    for c in range(w):
                        if(edges_res[f,c]==255):
                            aS+=1
                for f in range(int(h)):
                    for c in range(int(w/2)):
                        if(edges_res[f,c]==255):
                            aL+=1
                for f in range(h):
                    for c in range(int(w/2),int(w)):
                        if(edges_res[f,c]==255):
                            aR+=1
                aT=np.count_nonzero(edges_res)
                Comp = aT/float(p*p)
                
                VectorCarac = np.array([Hu[0][0],Hu[1][0],Hu[2][0]])
                print(VectorCarac)
                sample= VectorCarac.reshape(1,-1)
                mlp= joblib.load('model_mlp.pkl')
                knn=joblib.load('Modelos_Predict/knn')
                standard_scaler=joblib.load('model_scaler.pkl')
                X=standard_scaler.transform(sample)
                predMLP=mlp.predict(X)
                predKNN=knn.predict(X)
                if int(j)==int(predMLP[0]):
                    contMLP+=1
                if int(j)==int(predKNN[0]):
                    contKNN+=1
                print ('MLP:\t',int(predMLP[0]))
                print ('KNN:\t',int(predKNN[0]))
                print ('----------------------------------\t')                
cv2.destroyAllWindows()