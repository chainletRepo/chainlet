import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 150000, "display.max_columns", 48)


allData = "../data/allData.csv"
allData_df = pd.read_csv(allData, sep=",", header=0)
allData_df = allData_df.drop(["Unnamed: 0"],axis=1)

dat = allData_df[allData_df.columns[5:53]].apply(pd.to_numeric)
vals = dat.idxmax(axis="columns")
vals.hist(density=True)
plt.grid(visible=None)
plt.xticks(rotation=90)
plt.savefig('../figures/mostDominantOrbits.png')
plt.show()

# row normalize data
dat_norm = dat.div(dat.sum(axis=1), axis=0)
# sum columns
dat_norm_columns = dat_norm.sum(axis=0, skipna=True)
dat_norm_columns = dat_norm_columns/len(dat_norm.index)
dat_norm_columns.plot.bar()
plt.savefig('../figures/allOrbitPercent.png')
plt.show()

white = allData_df[allData_df["label"] == "white"]
virus = allData_df[allData_df["label"] != "white"]

# white orbits
white=white[white.columns[5:52]].apply(pd.to_numeric)
white_norm = white.div(white.sum(axis=1), axis=0)

# sum columns
white_norm_columns = white_norm.sum(axis=0, skipna=True)
white_norm_columns = white_norm_columns/len(white_norm.index)
white_norm_columns.plot.bar()
plt.savefig('../figures/whiteOrbitPerc.png')
plt.show()

# virus orbits
virus = virus[virus.columns[5:52]].apply(pd.to_numeric)
virus_norm = virus.div(virus.sum(axis=1), axis=0)

# sum columns
virus_norm_columns = virus_norm.sum(axis=0, skipna=True)
virus_norm_columns = virus_norm_columns/len(virus_norm.index)
virus_norm_columns.plot.bar()
plt.savefig('../figures/virusOrbitPerc.png')
plt.show()