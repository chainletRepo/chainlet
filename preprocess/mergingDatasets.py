import pandas as pd
import datetime

# Merge orbit_df and heist_df on address and date column
# Reading orbits file
# fields = ["year", "day", "address", "o0","o1","o2","o3","o4","o5","o6","o7","o8","o9","o10","o11","o12","o13","o14","o15","o16","o17","o18","o19","o20","o21","o22","o23","o24","o25","o26","o27","o28","o29","o30","o31","o32","o33","o34","o35","o36","o37","o38","o39","o40","o41","o42","o43","o44","o45","o46","o47",];
articleOrbits = "../rawData/articleOrbits.csv"
orbits_df = pd.read_csv(articleOrbits, sep="\t", header=0)
print(orbits_df.shape)
orbits_df['date'] = pd.to_datetime(orbits_df['year'].astype(str) + '-' + orbits_df['day'].astype(str), format='%Y-%j')
# Convert 'year', 'month' and 'day' column to 'date' column
# orbits_df['date'] = pd.to_datetime(orbits_df['year'].astype(str) + '-' +  orbits_df['month'].astype(str) + '-' + orbits_df['day'].
#                                    astype(str), format='%Y-%m-%j')

# Reading heist file
HeistData = "../rawData/BitcoinHeistData.csv"
heist_df = pd.read_csv(HeistData, sep=",", header=0)
print(heist_df.shape)

# Convert 'year' and 'day' column to 'date' column
heist_df['date'] = pd.to_datetime(heist_df['year'].astype(str) + '-' + heist_df['day'].astype(str), format='%Y-%j')
heist_df = heist_df.drop(["year", "day"], axis=1)

merged_df = orbits_df.merge(heist_df, left_on=['address','date'], right_on=['address', 'date'])
print(merged_df.shape)
merged_df.to_csv('../data/merged.csv')


# Concatenating merged file (ransomware and white addresses) with darknet file (darknet market addresses)
# merged = "../data/merged.csv"
# merged_df = pd.read_csv(merged, sep=",", header=0)


# dark = "../data/darkData.csv"
# dark_df = pd.read_csv(dark, sep=",", header=0)
#
# dark_df['year'] = dark_df['year'].astype(int)
# dark_df['month'] = dark_df['month'].astype(int)
# dark_df['day'] = dark_df['day'].astype(int)
#
# # Convert 'year', 'month' and 'day' column to 'date' column
# dark_df['date'] = pd.to_datetime(dark_df['year'].astype(str) + '-' +  dark_df['month'].astype(str) + '-' + dark_df['day'].
#                                                     astype(str), format='%Y-%m-%j')
#
# dark_df = dark_df.drop(["Unnamed: 0"], axis=1)
# allData = pd.concat([dark_df, merged_df], ignore_index=True, axis = 0)
#
# # Replace all different ransomware labels to "ransomware"
# allData.replace(['montrealCryptoLocker','montrealNoobCrypt','montrealDMALocker','paduaCryptoWall', 'montrealCryptoTorLocker2015',
#                     'montrealSamSam','montrealGlobeImposter','princetonCerber','montrealDMALockerv3','montrealGlobe',
#                     'montrealAPT','princetonLocky', "montrealCryptConsole", "montrealGlobev3", "montrealVenusLocker", "montrealXLockerv5.0"],'ransomware', inplace=True)
# print(allData.info())
#
# allData['date'] = pd.to_datetime(allData['date']).dt.date
#
# print(allData.info())
# allData = allData.drop(["Unnamed: 0"], axis=1)
#
# # Reindex 'date' column
# date_col = allData.pop('date')
# allData.insert(1, 'date', date_col)
#
# allData.to_csv('../data/allData.csv')