import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dalex as dx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn import metrics
from sklearn.inspection import permutation_importance
import numpy as np
import random

random.seed(10)

merged = '../data/merged.csv'
df = pd.read_csv(merged, usecols=['address','o3','o5','o7','o9','o12','o15','o18','o20','o22','o24','o27','o30','o33','o36','o39','o42','o44','o46','label'], sep=",", header=0)

df["label"].replace({"montrealAPT":"virus","princetonLocky":"virus","montrealCryptoLocker": "virus", "montrealNoobCrypt": "virus",'montrealDMALocker':'virus','paduaCryptoWall':'virus','montrealCryptoTorLocker2015':'virus','montrealSamSam':'virus','montrealGlobeImposter':'virus','princetonCerber':'virus','montrealDMALockerv3':'virus','montrealGlobe':'virus'}, inplace=True)

# Create dataframe for active orbits
df = df[(df.o3 !=0) |(df.o5 !=0) |(df.o7 !=0) |(df.o9 !=0) |(df.o12 !=0) |(df.o15 !=0) |(df.o18 !=0) |(df.o20 !=0) |(df.o22 !=0) |(df.o24 !=0) |(df.o27 !=0) |(df.o30 !=0) |(df.o33 !=0) |(df.o36 !=0) |(df.o39 !=0) |(df.o42 !=0) |(df.o44 !=0) |(df.o46 !=0)]

# print(df.sum(axis=0))

X = df[df.columns[1:19]]  # Features
y = df['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

######################
#random forest
clf = RandomForestClassifier(n_estimators=300, max_features=13)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Random Forest Accuracy - Active Orbits: ", metrics.accuracy_score(y_test, y_pred))
accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("Random Forest Cross Validation - Active Orbits: ", accuracies.mean())
print("Random Forest ROC_AUC_Score - Active Orbits: ", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
plot_precision_recall_curve(clf, X_test, y_test, name='Random forest - Active Orbits')
plt.savefig('figures/Prec_Active.png')
print("Confusion Matrix - Active Orbits")
print(confusion_matrix(y_test, y_pred))

#Shapely
explainer = dx.Explainer(clf, X, y, label = "Random Forest")
observation = df[df['label'] == 'virus'].sample()
observation = observation.drop(['address','label'],axis = 1)
print(observation)

shap_values = explainer.predict_parts(new_observation = observation, type = "shap",B=10)
fig = shap_values.plot(bar_width = 16)
