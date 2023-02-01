from itertools import cycle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

dataset = "../data/merged.csv"

df = pd.read_csv(dataset, sep=",", header=0)

df.replace(['paduaCryptoWall','montrealCryptoLocker','princetonCerber','princetonLocky', 'montrealCryptXXX',
                    'montrealDMALockerv3','montrealNoobCrypt','montrealDMALocker','montrealSamSam','montrealCryptoTorLocker2015',
                    'montrealGlobev3','montrealGlobe', "montrealWannaCry", "montrealGlobeImposter", "montrealRazy",
            "montrealAPT","montrealFlyper","montrealCryptConsole","montrealXTPLocker","paduaKeRanger","montrealXLockerv5.0",
            "montrealEDA2","montrealVenusLocker","montrealJigSaw","paduaJigsaw","montrealXLocker","montrealComradeCircle"],'ransomware', inplace=True)


df = df.rename(columns={'label_x': 'label'})
df = df.drop(columns=['label_y'])
df = df.drop(columns=['date'])
print(df['label'].value_counts())
X = df.iloc[:, 5:61]  # you need to change this column selection based on which features to use
print(X)

# Fit and transform the 'label' column
y = LabelBinarizer().fit_transform(df.label.to_numpy())
n_classes = y.shape[1]
names = df.label.unique()
print(names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

classifier = RandomForestClassifier(n_estimators=300)
# ovr = OneVsRestClassifier(classifier)

classifier.fit(X_train, y_train)
y_score = classifier.predict(X_test)

for i, classifier in enumerate(classifier.estimators_):
    print("Class", i)
    feature_importances = classifier.feature_importances_
    # Get the feature names
    feature_names = list(X.columns)
    # Create a DataFrame with the feature importances
    df = pd.DataFrame(data={'feature_name': feature_names, 'importance': feature_importances})
    # Save the DataFrame to a CSV file
    df.to_csv("../results/feature_importances.csv", index=False)


# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# colors = cycle(['blue', 'red'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#                    ''.format(names[i], roc_auc[i]))

# auc_results = {}
# for i in range(n_classes):
#     auc_results[names[i]] = roc_auc[i]

# Compute the AUC on the test set for both classes
# y_pred = classifier.predict_proba(X_test)
# positive_class_index = 1
# positive_class_auc = roc_auc_score(y_test, y_pred[:, positive_class_index])
# negative_class_index = 0
# negative_class_auc = roc_auc_score(y_test, y_pred[:, negative_class_index])
# print("Class labels:", classifier.classes_)
# print("AUC for positive class: ", positive_class_auc)
# print("AUC for negative class: ", negative_class_auc)
y_test_int = y_test.replace({'ransomware': 1, 'white': 0})
auc_lr = roc_auc_score(y_test_int, y_score)
print(auc_lr)
# Create a DataFrame from the dictionary
# auc_df = pd.DataFrame.from_dict(auc, orient='index')
# auc.to_csv('../results/auc.csv', mode='a', header=False)

# plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
# plt.xlim([-0.05, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(df.shape)
# plt.legend(loc="lower right")
# plt.show()
# plt.savefig('../results/all_features.png', bbox_inches='tight')
