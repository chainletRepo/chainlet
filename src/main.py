import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import shap
import dalex as dx
import random

random.seed(10)

merged = "merged.csv"
merged_df = pd.read_csv(merged, sep=",", header=0)
merged_df = merged_df.drop(["Unnamed: 0"],axis=1)
merged_df["label"].replace({"montrealAPT":"virus","princetonLocky":"virus","montrealCryptoLocker": "virus", "montrealNoobCrypt": "virus",'montrealDMALocker':'virus','paduaCryptoWall':'virus','montrealCryptoTorLocker2015':'virus','montrealSamSam':'virus','montrealGlobeImposter':'virus','princetonCerber':'virus','montrealDMALockerv3':'virus','montrealGlobe':'virus'}, inplace=True)

X = merged_df[merged_df.columns[4:52]]  # Features
y = merged_df['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

######################
# random forest
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, y_pred))
accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("Random Forest Cross Validation: ", accuracies.mean())
print("Random Forest ROC_AUC_Score: ", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
plot_precision_recall_curve(clf, X_test, y_test, name='Random forest')

print(confusion_matrix(y_test, y_pred))

# feature importance
feature_names = list(X.columns)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std)

#Shapely
explainer = dx.Explainer(clf, X, y, label = "Random Forest")
observation = merged_df[merged_df['label'] == 'virus'].sample()
observation = observation.drop(['address','label'],axis = 1)
shap_values = explainer.predict_parts(new_observation = observation, type = "shap",B=10)

fig = shap_values.plot(bar_width = 16)
plt.savefig('figures/shap.png')

######################
# xgboost
xg_clf = xgb.XGBClassifier(n_estimators = 300)
xg_clf.fit(X_train,y_train)
preds_xgb = xg_clf.predict(X_test)

print("cross validation score")
scores = cross_val_score(xg_clf,X,y,cv=5)
print(scores.mean())
print("xgboost confusion matrix")
print(confusion_matrix(y_test,preds_xgb))

accuracy = accuracy_score(y_test, preds_xgb)
print("xgboost accuracy")
print(accuracy)

xgb.plot_importance(xg_clf)
plt.rcParams['figure.figsize'] = [5, 5]
plt.savefig('figures/xgb_importance.png')


