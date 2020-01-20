import numpy as np
import pandas as pd

dataset = pd.read_csv('donations.csv')#Diavazei to arxeio
X = dataset.iloc[:,:].values  #Oles tis grammes & stiles ektos apo to state

#Diagrafi ta String apto .csv
X = np.delete(X, 9, 1) #y values
X = np.delete(X, 7, 1) #date
X = np.delete(X, 5, 1) #date
X = np.delete(X, 1, 1) #name

X = X.astype(dtype="float") #Metatrepei se float oti exei apominei mesa

Y = dataset.iloc[:,9].values  #Pernei to state
Y = Y.astype(dtype="float")   #Metatrepei se float ta states

#Opou uparxi h timi 0 ston pinaka X tin antikatasti me tin mesi timi tis stilis
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


ALL_Scores=[] #Apothikeuei ola ta Score & Error se mia lista
for i in range(1,1001): #Ektelei tin loopa 1000 fores
    #print(i,"η φορά:")
    
    #Xorizei ta dedomena se training kai test
    #pernei random to 25% tou arxeiou gia test
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

    #Ekpaideuei to neuroniko meso Linear Regression model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, Y_train)
    #print("\n\nTest Τιμές Μισθών:\n",Y_test)
    
    Y_pred = model.predict(X_test)
    #print("\n\nPredicted Τιμές Μισθών:\n",Y_pred)
    
    #Ypolozigi to Accuracy & to Meso Tetragoniko Sfalma metaksi Y_test Y_pred
    from sklearn.metrics import roc_auc_score,mean_squared_error
    score = roc_auc_score(Y_test, Y_pred)
    error = mean_squared_error(Y_test, Y_pred)
    #print("\n\n\nΤο Accuracy μεταξύ Test και Prediction Τιμών: ",score*100,'%')
    #print("\n\n\nΤο Error μεταξύ Test και Prediction Τιμών: ",error)
    ALL_Scores.append([score*100,error])
    #print("===========================================================================")

#Diagrafi ta bracket apo to list
def listWithoutBrackets(ALL_Scores):
    return str(ALL_Scores).replace('[','').replace('],','\n').replace(']','')

print("\nΌλα τα Accuracy Score:\n",listWithoutBrackets(ALL_Scores))
print("\nΜεγαλύτερο Accuracy Score & Error:",listWithoutBrackets(max(*ALL_Scores)))

