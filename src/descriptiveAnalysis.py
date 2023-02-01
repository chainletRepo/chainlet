import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


allData = "../data/allData.csv"
allData_df = pd.read_csv(allData, sep=',', header=0)
allData_df = allData_df.drop("Unnamed: 0", axis = 1)

# print(allData_df.info())
# print(allData_df.shape)

start = "2011-1-1"
end = "2017-12-31"
counts = allData_df.groupby('date')['address'].size()
# counts.to_csv('../data/addressCounts.csv')

# average number of values in column 'date' for each unique value in column 'address'
means = allData_df.groupby('date')['address'].size().mean()
# print(means)

# number of addresses in each year
addressYear = allData_df.groupby('year')['address'].size()
# print(addressYear)
plt.figure(figsize = (12,8))
addressYear.plot(kind='bar')
# plt.show()

yearUniqueLabel = allData_df.groupby("year")['label'].value_counts()
print(yearUniqueLabel)


# total number of unique values in column 'B'
total = allData_df['address'].nunique()
# print(total)

print(allData_df['label'].value_counts())



