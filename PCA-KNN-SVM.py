'''
Digit recognition for the Kaggle challenge http://www.kaggle.com/c/digit-recognizer
'''

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import svm

# Loading the dataset in memory
data = pd.read_csv("data/train.csv")
data = data.values

test = pd.read_csv("data/test.csv")
test = test.values

# Separating the labels from the training set
train = data[:, 1:]
labels = data[:, :1]

def KNN(tr,la,te):
    # Create a classifier, KNN
    estimators = [('reduce_dim', PCA(n_components = 100)), ('Knn', KNeighborsClassifier(weights = 'distance', n_neighbors=5, p=3))]
    knn = Pipeline(estimators)
    # The learning is done on the first half of the dataset
    knn.fit(tr, la)

    # Now predict the value of the digit in test:
    predicted = knn.predict(te)
    predicted = pd.DataFrame(predicted)
    predicted['ImageId'] = predicted.index + 1
    predicted = predicted[['ImageId', 0]]
    predicted.columns = ['ImageId', 'Label']

    predicted.to_csv('knn_pred.csv', index=False)

def SVM(tr,la,te):
    # Create a classifier, SVM
    Svm = svm.SVC(gamma=0.1, kernel='poly')
    # The learning is done on the first half of the dataset
    Svm.fit(tr, la)

    # Now predict the value of the digit in test :
    predicted = Svm.predict(te)

    predicted = pd.DataFrame(predicted)

    predicted['ImageId'] = predicted.index + 1
    predicted = predicted[['ImageId', 0]]
    predicted.columns = ['ImageId', 'Label']

    predicted.to_csv('svm_pred.csv', index=False)

KNN(train,labels,test)
SVM(train,labels,test)