import pandas as pd

dataset = "../rawData/v2articleOrbits.csv"

df = pd.read_csv(dataset, sep="\t", header=0)
print(df.head())
df['pattern'] = df[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47']].apply(lambda x: ''.join(x.astype(str)), axis=1)
pattern = ['000000000000011000000000000000000000000000000000','010000000000000000000000000000000000000000000000',
           '010000000000010000000000000000000000000000000000','000000000100011000000000000000000000000000000000',
           '000000000001110000000000000000000000000000000000']
print("1")
# Filter for rows where the label is 'white' and pattern is 'your_desired_pattern'
# df_filtered = df.query("label == 'white' and pattern in @pattern")
counts_per_year = pd.DataFrame(columns=['year', 'pattern', 'count'])

for p in pattern:
    df_filtered = df.loc[(df['label'] == 'white') & (df['pattern'] == p)]
    print("2")
    grouped_df = df_filtered.groupby('year').size().reset_index(name='count')
    grouped_df['pattern'] = p
    counts_per_year = counts_per_year.append(grouped_df, ignore_index=True)

counts_per_year.to_csv("../results/counts_per_year.csv", header=True, index=False)


