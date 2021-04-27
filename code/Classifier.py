# -*- coding: utf-8 -*-
"""

@author: M
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#prepare data
data = pd.read_csv('C:\\Users\\M\\Downloads\\diabetes.csv')
X = data[data.columns.tolist()[:-1]].to_numpy()
y = data['Outcome'].to_numpy()

#Classifier
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('kn', KNeighborsClassifier()),
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=MLPClassifier(random_state=42 , hidden_layer_sizes = 128)
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42,train_size = 0.7
)
print(clf.fit(X_train, y_train).score(X_test, y_test))

#vis
emd = PCA(n_components=2).fit_transform(X)
plt.scatter(emd[:,0] , emd[:,1] , c = y)
plt.title('PCA embedding 2d')
plt.show()

emd = TSNE(n_components=2).fit_transform(X)
plt.scatter(emd[:,0] , emd[:,1] , c = y)
plt.title('TSNE embedding 2d')
plt.show()

#3D vis
from mpl_toolkits.mplot3d import Axes3D

ps = PCA(n_components=3).fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
ax.scatter(ps[:,0] , ps[:,1] , ps[: , 2] , c = y)
plt.title('PCA embedding 3d')
plt.show()


ps = TSNE(n_components=3).fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
ax.scatter(ps[:,0] , ps[:,1] , ps[: , 2] , c = y)
plt.title('TSNE embedding 3d')
plt.show()

