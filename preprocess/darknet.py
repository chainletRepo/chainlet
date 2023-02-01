import numpy as np
import pandas as pd
import os

# # Load darknet files for each day, merge it with merged file to get the orbits of the address
# merged = "../data/merged.csv"
# merged_df = pd.read_csv(merged, sep=",", header=0)
#
# for file in os.listdir('../rawData/darknetData/bad_addr'):
#     if file.endswith(".npy"):
#         d = np.load("../rawData/darknetData/bad_addr/" + file)
#         print(file + " " + str(len(d)))
#         addr_dir = np.load('../rawData/darknetData/addr_dir.npy')
#         d_addresses = addr_dir[d]
#     # np.save('days_addr/'+ file, d_addresses)
#     df = pd.DataFrame(d_addresses, columns=['address'])
#     # print(d_addresses)
#     day_df = df.merge(merged_df, left_on="address", right_on='address')
#     df = df.drop_duplicates(subset=['address'], keep="first")
#
#     # Concatenate all days into one dataframe
#     df = pd.concat([df, pd.DataFrame(columns = ['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10',
#                                    'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21',
#                                    'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32',
#                                    'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43',
#                                    'o44', 'o45', 'o46', 'o47'])])
#
#     df['label'] = 'darknet'
#     df.to_csv('../rawData/darknetData/days_addr/'+ file.rpartition('.')[0])


darknet = '../data/darkData.csv'
darknet_df = pd.read_csv(darknet, sep=",", header=0)

# Extracting darknet market addresses
darknetAddress_df = darknet_df['address']
# darknetAddress_df.to_csv('../data/darknetAddresses.csv')

darknetAddress_df = darknetAddress_df.drop_duplicates()
darknetAddress_df.to_csv('../data/darknetAddresses.csv',index=False)
