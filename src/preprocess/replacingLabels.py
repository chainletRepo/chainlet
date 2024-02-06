import pandas as pd

# data = pd.read_csv("../rawData/V2articleOrbits.csv", sep='\t', header=0, index_col=False)
# print(data.value_counts('label'))
#
# data.replace(['montrealCryptoLocker','montrealNoobCrypt','montrealDMALocker','paduaCryptoWall', 'montrealCryptoTorLocker2015',
#                     'montrealSamSam','montrealGlobeImposter','princetonCerber','montrealDMALockerv3','montrealGlobe',
#                     'montrealAPT','princetonLocky', "montrealCryptConsole", "montrealGlobev3", "montrealVenusLocker", "montrealXLockerv5.0"],'ransomware', inplace=True)
# data.drop(data[data['label'].isin(['EXCHANGE', 'SERVICE', 'GAMBLING', 'MINING'])].index, inplace=True)
# print(data.value_counts('label'))
# print(data.shape)
#
# data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['dayOfYear'].astype(str), format='%Y-%j')
# data.drop(["year", "dayOfYear"], axis=1)
# data.to_csv('../data/newData.csv')

df = pd.read_csv('../data/newData.csv', sep=',', header=0)
df.reset_index(inplace=True, drop=True)
# df.drop(["Unnamed: 0"], axis=1)

df.to_csv('../data/newData_edited.csv', index=False)