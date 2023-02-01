import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

merged = "merged.csv"
merged_df = pd.read_csv(merged, sep=",", header=0)
merged_df = merged_df.drop(["Unnamed: 0"],axis=1)

X = merged_df[merged_df.columns[4:52]]  # Features
y = merged_df['label']  # Labels

# Scaling row-wise
a = np.array(X)
scaler = StandardScaler()
scaled_row = pd.DataFrame(scaler.fit_transform(a.T))
scaled_row = scaled_row.T
X = scaled_row[scaled_row.columns]  # Features

# Running PCA
pca = PCA(n_components=25)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# Explained variance
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# Screeplot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()