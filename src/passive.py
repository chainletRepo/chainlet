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

merged = 'merged.csv'
df = pd.read_csv(merged, usecols=['address','o0','o1','o2','o4','o6','o8','o10','o11','o13','o14','o16','o17','o19','o21','o23','o25','o26','o28','o29','o31','o32','o34','o35','o37','o38','o40','o41','o43','o45','o47','label'], sep=",", header=0)

df["label"].replace({"montrealAPT":"virus","princetonLocky":"virus","montrealCryptoLocker": "virus", "montrealNoobCrypt": "virus",'montrealDMALocker':'virus','paduaCryptoWall':'virus','montrealCryptoTorLocker2015':'virus','montrealSamSam':'virus','montrealGlobeImposter':'virus','princetonCerber':'virus','montrealDMALockerv3':'virus','montrealGlobe':'virus'}, inplace=True)

df = df[(df.o0 != 0)|(df.o1 != 0)|(df.o2 != 0)|(df.o4 != 0) | (df.o6 != 0) | (df.o8 != 0) | (df.o10 != 0) | (df.o11 != 0)| (df.o13 != 0) | (df.o14 != 0) | (df.o16 != 0)| (df.o17 != 0) | (df.o19 != 0) | (df.o21 != 0) | (df.o23 != 0) | (df.o25 != 0) | (df.o26 != 0) | (df.o28 != 0) | (df.o29 != 0) | (df.o31 != 0) | (df.o32 != 0) | (df.o34 != 0) | (df.o35 != 0) | (df.o37 != 0) | (df.o38 != 0) | (df.o40 != 0) | (df.o41 != 0) | (df.o43 != 0) | (df.o45 != 0) | (df.o47 != 0)]

X = df[df.columns[1:31]]  # Features
y = df['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

######################
# random forest
clf = RandomForestClassifier(n_estimators=300, max_features=13)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Random Forest Accuracy - Passive Orbits: ", metrics.accuracy_score(y_test, y_pred))
# accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
# print("Random Forest Cross Validation - Passive Orbits: ", accuracies.mean())
print("Random Forest ROC_AUC_Score - Passive Orbits: ", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
# plot_precision_recall_curve(clf, X_test, y_test, name='Random forest - Passive Orbits')
# plt.savefig('figures/Prec_Passive.png')
print("Confusion Matrix - Passive Orbits")
print(confusion_matrix(y_test, y_pred))

#Shapely
explainer = dx.Explainer(clf, X, y, label = "Random Forest")
observation = df[df['label'] == 'virus'].sample()
observation = observation.drop(['address','label'],axis = 1)
print(observation)
shap_values = explainer.predict_parts(new_observation = observation, type = "shap",B=10)

fig = shap_values.plot(bar_width = 16)
