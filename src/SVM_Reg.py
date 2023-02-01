import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import random

random.seed(10)


merged = 'merged.csv'
df = pd.read_csv(merged, usecols=['address','o3','o5','o7','o9','o12','o15','o18','o20','o22','o24','o27','o30','o33','o36','o39','o42','o44','o46','label'], sep=",", header=0)

df["label"].replace({"montrealAPT":"virus","princetonLocky":"virus","montrealCryptoLocker": "virus", "montrealNoobCrypt": "virus",'montrealDMALocker':'virus','paduaCryptoWall':'virus','montrealCryptoTorLocker2015':'virus','montrealSamSam':'virus','montrealGlobeImposter':'virus','princetonCerber':'virus','montrealDMALockerv3':'virus','montrealGlobe':'virus'}, inplace=True)
df["label"].replace({"virus":1,"white":0}, inplace=True)

df = df[(df.o3 !=0) |(df.o5 !=0) |(df.o7 !=0) |(df.o9 !=0) |(df.o12 !=0) |(df.o15 !=0) |(df.o18 !=0) |(df.o20 !=0) |(df.o22 !=0) |(df.o24 !=0) |(df.o27 !=0) |(df.o30 !=0) |(df.o33 !=0) |(df.o36 !=0) |(df.o39 !=0) |(df.o42 !=0) |(df.o44 !=0) |(df.o46 !=0)]#

X = df[df.columns[1:19]]  # Features
y = df['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#################
# SVM RBF with grid search
svm_rbf = SVC()
svm_rbf.fit(X_train,y_train)

c = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 5, 7,15,30]

for i in c:
    param_grid = {'C': [i], 'gamma': ["auto"]}
    grid = GridSearchCV(SVC(),param_grid)
    grid.fit(X_train,y_train)
    grid_predictions = grid.predict(X_test)
    print("confusion matrix with " + str(i))
    print(confusion_matrix(y_test,grid_predictions))

##################
# SVM Linear with grid search
svm_linear = SVC()
svm_linear.fit(X_train,y_train)

c = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 5, 7,15,30]

for i in c:
    param_grid = {'kernel':'linear', 'C': [i], 'gamma': ["auto"]}
    grid = GridSearchCV(SVC(),param_grid)
    grid.fit(X_train,y_train)
    grid_predictions = grid.predict(X_test)
    print("confusion matrix with " + str(i))
    print(confusion_matrix(y_test,grid_predictions))

# #####################
# SVM with linear kernel and rbf
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
y_pred_linear = clf_linear.predict(X_test)

print("SVM with Linear kernel Accuracy: ",metrics.accuracy_score(y_test, y_pred_linear))
print(confusion_matrix(y_test, y_pred_linear))

# SVM with radial kernel
clf_radial = svm.SVC(kernel='rbf')
clf_radial.fit(X_train, y_train)
y_pred_radial = clf_radial.predict(X_test)

print("SVM with radial kernel Accuracy: ",metrics.accuracy_score(y_test, y_pred_radial))
print(confusion_matrix(y_test, y_pred_radial))

######################
# linear regression
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)
y_pred = linear_reg.predict(X_test)

print("Coefficients: \n", linear_reg.coef_)
print("Mean squared error: %.6f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print("Intercept: %6f" % linear_reg.intercept_)
print(linear_reg.score(X_test, y_test))

######################
# logistic regression
logistic_reg = LogisticRegression(random_state=0).fit(X_train,y_train)
y_pred = logistic_reg.predict(X_test)
print(logistic_reg.score(X_test, y_test))
print("Coefficients: \n", logistic_reg.coef_)
print("Mean squared error: %.6f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print("Intercept: %6f" % logistic_reg.intercept_)
