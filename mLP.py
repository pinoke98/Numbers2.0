from __future__ import print_function, division
import numpy as np
import cv2
import xlrd
from time import time
import pandas as pd
#MULTI LINE PERCEPTRON
#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

#from sklearn.model_selection import cross_val_score

data = pd.read_excel("DatosTrain.xlsx")


def load_xlsx(d):
    #print(d)
    d=np.array(d).astype(np.float64)
    d=d.transpose()
    y=d[0]
    x=d[1:4].transpose()
    return x,y
    
##### Inicio del programa ######
if __name__ == '__main__':
    aI=100
    accuracyVector=np.zeros(aI)
    for a in range(aI):
        t0 = time()
        
        # Cargar datos desde un archivo .xlsx
        # la funci�n retornar� el n�mero de muestras obtenidas y su respectiva clase
        X, Y = load_xlsx(data)
        #print (len(X),len(Y))
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        # Se separan los datos: un % para el entrenamiento del modelo y otro
        # para el test
        samples_train, samples_test, responses_train, responses_test = \
                    train_test_split(X, Y, test_size = 0.2)

        mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100,100), max_iter=1000000, tol=0.00001)
        mlp.fit(samples_train, responses_train)    
        response_pred = mlp.predict(samples_test)
        accuracyVector[a]=mlp.score(samples_test, responses_test)*100.0
        print ("Iteration",a+1,"with accuracy score of", mlp.score(samples_test, responses_test)*100.0)
        if mlp.score(samples_test, responses_test)*100==100:
            break
    joblib.dump(standard_scaler,'model_scaler.pkl')    
    joblib.dump(mlp,'model_mlp.pkl')
    prom=sum(accuracyVector)/aI
    print ("Average accuracy ",prom)
    print ("\n")
    print("done in %0.16fs" % (time() - t0))

