import pandas as pd
import csv

df = pd.read_csv('../data/newData_edited.csv', sep=',', header=0)

df.replace(['montrealCryptoLocker','montrealNoobCrypt','montrealDMALocker','paduaCryptoWall', 'montrealCryptoTorLocker2015',
                    'montrealSamSam','montrealGlobeImposter','princetonCerber','montrealDMALockerv3','montrealGlobe',
                    'montrealAPT','princetonLocky', "montrealCryptConsole", "montrealGlobev3", "montrealVenusLocker", "montrealXLockerv5.0"],'ransomware', inplace=True)

df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year

df['pattern'] = df[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47']].apply(lambda x: ''.join(x.astype(str)+"\t"), axis=1)
# Group rows by year and month and apply value_counts to find most frequent pattern in each group
# most_frequent_pattern = df.groupby(['year', 'month','label'])['pattern'].apply(lambda x: x.value_counts().index[0])
# most_frequent_pattern = most_frequent_pattern.rename(columns={most_frequent_pattern.columns[-1]: "pattern"})
# most_frequent_pattern[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17',
#     'o18', 'o19', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32', 'o33', 'o34',
#     'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47']]=most_frequent_pattern.pattern.str.split('\t',expand=True)



most_frequent_pattern = df.groupby(['year', 'month','label']).apply(lambda x: x.loc[x.loc[:,'pattern'].value_counts().idxmax()])
most_frequent_pattern = pd.DataFrame(most_frequent_pattern)
# Create a new dataframe that contains only the 'o0' to 'o47' columns
df_pattern = df[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47']]

# Concatenate the most_frequent_patterns with df_pattern
most_frequent_patterns = pd.concat([most_frequent_pattern, df_pattern], axis=1)
print(most_frequent_pattern)
# most_frequent_pattern = most_frequent_pattern.reset_index()

# # Split the "pattern" column into individual characters
# pattern_split = most_frequent_pattern.apply(lambda x: pd.Series(list(x)))
# most_frequent_pattern = pd.concat([most_frequent_pattern, pattern_split], axis=1)
# print(pattern_split.head())
# # Rename the columns
# pattern_split.columns = ["o" + str(i) for i in range(48)]
#
# # Concatenate the resulting dataframe with the original dataframe
# most_frequent_pattern = pd.concat([most_frequent_pattern, pattern_split], axis=1)
#
# # Drop the "pattern" column
# most_frequent_pattern = most_frequent_pattern.drop(columns=["pattern"])
#
# # Print the resulting dataframe
# print(most_frequent_pattern)
# most_frequent_pattern.to_csv('../data/freqOrbitPatterns3.csv')

# # Group the data by year, month, and label
# grouped = df.groupby(['year', 'month', 'label'])
#
#
# def find_most_frequent_pattern(group):
#     # Create a new column that concatenates the values of the columns o0 to o47
#     group['pattern'] = group.iloc[:,5:53].apply(lambda x: ''.join(x.astype(str)), axis=1)
#     # Find the most frequent pattern
#     most_frequent_pattern = group['pattern'].mode().values[0]
#     # Convert the pattern to a list
#     pattern_list = list(most_frequent_pattern)
#     # Return the pattern list
#     return pattern_list
#
# #
# # Apply the find_most_frequent_pattern function to each group
# most_frequent_pattern = grouped.apply(find_most_frequent_pattern)


#
#
#
# # # Group rows by label and apply value_counts to find count of each pattern
# # label_pattern_count = df.groupby('label')['pattern'].apply(lambda x: x.value_counts()).reset_index(name='label_pattern_count')
# # print(label_pattern_count)
# # label_pattern_count = label_pattern_count.rename(columns={'level_1': 'pattern'})
# # # Merge pattern counts with original dataframe
# # df = pd.merge(df, label_pattern_count, on=['label', 'pattern'])
# #
# # Count frequency of each pattern
# # pattern_counts = df['pattern'].value_counts()
# #
# # # Convert pattern counts to dictionary
# # pattern_counts_dict = dict(pattern_counts)
# #
# # # Map dictionary to pattern column and assign result to new column
# # df['pattern_frequency'] = df['pattern'].map(pattern_counts_dict)
# #
# # df.to_csv('../data/frequency3.csv')
# # # print(df.head())
df = pd.read_csv('../data/frequency.csv', sep=',', header=0)

df[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17',
    'o18', 'o19', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32', 'o33', 'o34',
    'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47']]=split_by_n(1)

print(df.head())
