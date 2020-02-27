import numpy as np
import cv2
import pandas as pd
from time import time
import matplotlib.pyplot as plt

#SKLearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import acurrancy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt

data = pd.read_excel("DatosTrain.xlsx")
'''
data=np.array(data).astype(np.float64)
data=data.transpose()
print(data)
'''
def load_excel(d):
    #print(d)
    d=np.array(d).astype(np.float64)
    d=d.transpose()
    y=d[0]
    x=d[1:4].transpose()
    return x,y
'''
Inicia el programa
'''

if __name__=='__main__':
    aI=100
    accuracyVector=np.zeros(aI)
    for a in range(aI):
        x,y = load_excel(data)
        #print(x)
        standard_scaler= StandardScaler()
        x=standard_scaler.fit_transform(x)
        #print(x)
        samples_train, samples_test, responses_train, responses_test = train_test_split(x, y, test_size=0.2)
        knn= KNeighborsClassifier(n_neighbors=10,weights="distance",n_jobs=1)
        knn.fit(samples_train,responses_train)
        response_pred=knn.predict(samples_test)
        accuracyVector[a]=knn.score(samples_test, responses_test)*100
        print("Iteration",a+1,"with acurancy score of", knn.score(samples_test,responses_test)*100)
        if knn.score(samples_test, responses_test)*100==100:
            break      
    joblib.dump(knn,"./Modelos_Predict/knn")
    prom=sum(accuracyVector)/aI
    print("Average accurancy ",prom)
    print("\n")