import pandas as pd
import numpy as np

#import dataset
dataset = pd.read_csv('/content/Iris.csv')
dataset.head()

#dividing the x and y values
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

#dataset has character value so we use label encoder to change it to numeric value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset["Species"] = le.fit_transform(dataset["Species"])
y = dataset["Species"]

#Divide dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing GaussianNB from sklearn
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting the results
y_pred  =  classifier.predict(X_test)
print( y_pred )

#printing the actual values of y
print(y_test)

#print confusion matrix and accuracy of Naive Bayes Implementation
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)
