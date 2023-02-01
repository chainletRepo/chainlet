import pandas as pd

allData = "../data/allData.csv"
allData_df = pd.read_csv(allData, sep=',', header=0)
allData_df = allData_df.drop("Unnamed: 0", axis = 1)

huang = "C:/A1/PhD MB/Chainlet/BAClassifier-main/BAClassifier-main/dataset/address_behavior_dataset.csv"
huang_df = pd.read_csv(huang, sep=',', header=0)

# First, merge the two dataframes on the "address" column
df_merged = allData_df.merge(huang_df, on='address', how='inner')

# Next, use the `loc` method to select the rows in the merged dataframe where the label in df1 is "white"
df_update = df_merged.loc[df_merged['label_x'] == 'white']

# Finally, update the labels in df1 using the `loc` method and the 'label_y' column from the merged dataframe
allData_df.loc[allData_df['address'].isin(df_update['address']), 'label'] = df_update['label_y']

allData_df.to_csv("../data/allDataModified.csv")