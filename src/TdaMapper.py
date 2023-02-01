import pandas as pd
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
import sklearn
import os
import numpy as np
from csv import reader

pd.set_option('mode.chained_assignment', None)

orbit_count = 49
threashold_for_output = 30
abs_dir_path = "C:/Users/poupa/PycharmProjects/chainletorbits/results/"

all_df = pd.DataFrame(columns=['address', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10',
                                   'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21',
                                   'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32',
                                   'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43',
                                   'o44', 'o45', 'o46', 'o47','label'])
vec_df = pd.DataFrame()

HeistData = "BitcoinHeistData.csv"
heist_df = pd.read_csv(HeistData, sep=",", header=0)
heist_df = heist_df.drop(['year','day','length','weight','count','looped','neighbors','income'],axis=1)
heist_df["label"].replace({"montrealAPT": "virus", "princetonLocky": "virus", "montrealCryptoLocker": "virus",
                            "montrealNoobCrypt": "virus", 'montrealDMALocker': 'virus', 'paduaCryptoWall': 'virus',
                            'montrealCryptoTorLocker2015': 'virus', 'montrealSamSam': 'virus',
                            'montrealGlobeImposter': 'virus', 'princetonCerber': 'virus',
                            'montrealDMALockerv3': 'virus', 'montrealGlobe': 'virus'}, inplace=True)

heist_virus_df = heist_df[(heist_df.label == 'virus')]
#create results file
output_reductions = open("../output_reductions.txt", "a")
output_matrices = open("../output_matrices.txt", "a")
files = os.listdir("days_edited")
darknet_files = os.listdir("days_addr")

n = 0

for f in files:
    list = []
    with open("days_edited/" + f, 'r') as read_obj:
        csv_reader = reader(read_obj)
        rows = 0
        for row in csv_reader:
            rows += 1
            for item in row:
                data = item.split("\t")
                vector = np.array(
                    [data[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'white'])
                if data[0] in heist_virus_df['address'].values:
                    vector = np.array(
                        [data[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'virus'])
                for s in data:
                    if ':' in s:
                        ss = s.split(":")
                        vector[int(ss[0]) + 1] = ss[1]
                list.append(vector)


    vec_df = pd.DataFrame(list,
                          columns=['address', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10',
                                   'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20', 'o21',
                                   'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o30', 'o31', 'o32',
                                   'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o40', 'o41', 'o42', 'o43',
                                   'o44', 'o45', 'o46', 'o47','label'])
    n += 1
    darknet_df = pd.read_csv("days_addr/" + str(n), sep=",", header=0)
    vec_df = vec_df.append(darknet_df, ignore_index=True)
    all_df = all_df.append(vec_df, ignore_index=True)
    # vec_df = pd.read_csv("days_csv/" + f, sep=",", header=0)
    past_virus_df = all_df[(all_df.label == 'virus')]
    reduction_active_list = []
    reduction_virus_list = []
    vec_df_white = vec_df[(vec_df['label']) == 'white']
    vec_active_df = vec_df[(vec_df.label == 'white') & (
                (vec_df.o3 != 0) | (vec_df.o5 != 0) | (vec_df.o7 != 0) | (vec_df.o9 != 0) | (vec_df.o12 != 0) | (vec_df.o15 != 0) | (
                vec_df.o18 != 0) | (vec_df.o20 != 0) | (vec_df.o22 != 0) | (vec_df.o24 != 0) | (vec_df.o27 != 0) | (vec_df.o30 != 0) | (
                        vec_df.o33 != 0) | (vec_df.o36 != 0) | (vec_df.o39 != 0) | (vec_df.o42 != 0) | (vec_df.o44 != 0) | (
                        vec_df.o46 != 0))]

    white_of_day = len(vec_df_white.index)
    only_important = vec_active_df[
        (vec_active_df.o3 != 0) | (vec_active_df.o9 != 0) | (vec_active_df.o24 != 0) | (vec_active_df.o33 != 0) | (
                vec_active_df.o42 != 0)]

    num_addr_to_analyze = len(only_important.index)
    reduction_active = 1 - (num_addr_to_analyze / white_of_day)
    reduction_active_list.append(reduction_active)

    vec_virus_df = vec_df[(vec_df.label == 'virus')]
    vec_virus_df["label"].replace({"virus": "v2"}, inplace=True)
    if len(vec_virus_df.index) == 0:
        continue
    past_virus_df["label"].replace({"virus": "v1"}, inplace=True)
    viruses_of_day = len(vec_virus_df.index)
    active_viruses = vec_virus_df[
        (vec_virus_df.o3 != 0) | (vec_virus_df.o9 != 0) | (vec_virus_df.o24 != 0) | (vec_virus_df.o33 != 0) | (vec_virus_df.o42 != 0)]
    all_active_viruses = vec_virus_df[
        (vec_virus_df.o3 != 0) | (vec_virus_df.o5 != 0) | (vec_virus_df.o7 != 0) | (vec_virus_df.o9 != 0) | (vec_virus_df.o12 != 0) | (
                vec_virus_df.o15 != 0) |
        (vec_virus_df.o18 != 0) | (vec_virus_df.o20 != 0) | (vec_virus_df.o22 != 0) | (vec_virus_df.o24 != 0) | (
                    vec_virus_df.o27 != 0) | (
                vec_virus_df.o30 != 0) |
        (vec_virus_df.o33 != 0) | (vec_virus_df.o36 != 0) | (vec_virus_df.o39 != 0) | (vec_virus_df.o42 != 0) | (
                    vec_virus_df.o44 != 0) | (
                vec_virus_df.o46 != 0)]
    active_viruses_of_day = len(active_viruses.index)
    reduction_virus = 1 - (active_viruses_of_day / viruses_of_day)
    reduction_virus_list.append(reduction_virus)

    output_reductions.write(f[:4] + "\t" + f[5]+"\t"+f[6]+"\t" + f[8]+"\t"+f[9]+ "\t" + str(reduction_active) + "\t" + str(
        reduction_virus) + "\t" + str(white_of_day) + "\t" + str(viruses_of_day))
    output_reductions.write("\n")
    output_reductions.write(str(len(vec_df)) + "\t" + str(len(all_df)) + "\t" + str(len(past_virus_df)) + "\t" +
                            str(len(vec_df_white)) + "\t" + str(len(vec_active_df)) + "\t" + str(len(only_important)) +
                            "\t" + str(len(vec_virus_df)))

    m = (only_important, active_viruses, past_virus_df)
    merged_day = pd.concat(m, ignore_index=True)
    address = merged_day.address
    label = merged_day.label
    M2 = merged_day
    y = M2.label
    M = M2[M2.columns[1:orbit_count]].apply(pd.to_numeric)
    Xfilt = M
    cls = len(pd.unique(y))
    mapper = km.KeplerMapper()
    scaler = MinMaxScaler(feature_range=(0, 1))
    Xfilt = scaler.fit_transform(Xfilt)
    lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE(verbose=1))
    print(" mapper started with " + str(len(pd.DataFrame(Xfilt).index)) + " data points," + str(cls) + " clusters")
    graph = mapper.map(
        lens,
        Xfilt,
        clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
        # clusterer=DBSCAN(eps=3, min_samples=2),
        cover=km.Cover(n_cubes=10, perc_overlap=0.2)
    )
    print(" mapper ended")
    print(str(len(y)) + " " + str(len(Xfilt)))
    if active_viruses_of_day > threashold_for_output:
        html = mapper.visualize(
            graph,
            path_html="results/orbit%s.html" % f,
            title="orbit data",
            custom_tooltips=y.to_numpy())

    for key, cluster in graph['nodes'].items():

        address_labels = []
        for node in cluster:
            address_id = address[node]
            address_label = label[node]
            # print(address_id,address_label)
            address_labels.append(address_label)
        count_v1 = address_labels.count('v1')
        count_v2 = address_labels.count('v2')
        count_w = address_labels.count('white')
        if count_v1 > 0:
            count_v1_ = count_v1 - 1
        else:
            count_v1_ = count_v1

        if count_v2 > 0:
            count_v2_ = count_v2 - 1
        else:
            count_v2_ = count_v2

        if count_w > 0:
            count_w_ = count_w - 1
        else:
            count_w_ = count_w

        matrix = np.array(
            [[count_v1_, count_v2, count_w], [count_v1, count_v2_, count_w], [count_v1, count_v2, count_w_]])

        if (count_v1 != 0 or count_v2 != 0):
            output_matrices.write(key)
            output_matrices.write("\n")
            matrix_str = str(matrix)
            mat_no_brack = str(matrix).replace(' [', '').replace('[', '').replace(']', '')
            output_matrices.write(mat_no_brack)
            output_matrices.write("\n")





