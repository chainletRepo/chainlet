import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import sys
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import networkx as nx
from OrbitFeaturizer import OrbitFeaturizer
from torchmetrics.classification import BinaryAUROC
from os import listdir
from os.path import isfile, join

from sklearn.metrics import roc_auc_score

class ThisDataset(Dataset):
    def __init__(self, root, n_graphs,filepath, addressFile, sampleWhite=sys.maxsize, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.num_graphs=n_graphs
        self.test = test
        self.sample = sampleWhite
        self.addressFile = addressFile
        self.filename = filepath
        super(ThisDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            #return [f'data_test_{i}.pt' for i in list(self.data.index)]
            return [f'data_test_{i}.pt' for i in range(0,self.num_graphs)]
        else:
            #return [f'data_{i}.pt' for i in list(self.data.index)]
            return [f'data_{i}.pt' for i in range(0, self.num_graphs)]


    def download(self):
        pass
    def get_two_chainlet_optimized(self,node,G):
        ge = nx.single_source_shortest_path_length(G,node,cutoff=2)
        return G.subgraph(ge.keys())

    def get_two_chainlet(self,node,graph):
        subgraph = nx.ego_graph(graph, node, radius=2,undirected=True)
        return subgraph
    def process_labels(self,df):
        ransom_families = ["montrealComradeCircle", "montrealCryptXXX", "montrealFlyper", "montrealJigSaw",
                           "montrealWannaCry", "montrealXTPLocker", "montrealXLocker", "paduaJigsaw", "paduaKeRanger",
                           'montrealEDA2', 'montrealRazy', 'montrealCryptoLocker', 'montrealNoobCrypt',
                           'montrealDMALocker', 'paduaCryptoWall', 'montrealCryptoTorLocker2015',
                           'montrealSamSam', 'montrealGlobeImposter', 'princetonCerber', 'montrealDMALockerv3',
                           'montrealGlobe',
                           'montrealAPT', 'princetonLocky', "montrealCryptConsole", "montrealGlobev3",
                           "montrealVenusLocker", "montrealXLockerv5.0"]
        df['label'] = df['label'].apply(
            lambda x: 'ransomware' if x in ransom_families else x
        )
        return df
    def process(self):
        #graph files
        sys.exit("Files should have already been processed")
        mypath= self.raw_paths[0]
        graph_files = [os.path.join(mypath,f) for (mypath, dirnames, filenames) in os.walk(mypath) for f in filenames]
        label_map  = {'white': 0, 'ransomware': 1, 'darknet': 2}
        index =0

        for file in np.sort(graph_files):

            print( "processing file: ", file)
            data = pd.read_csv(file,delimiter='\t',header=None).reset_index(drop=True)
            data.columns =["source","target","weight"]
            df = pd.read_csv(self.addressFile,delimiter='\t',header=0)
            df = self.process_labels(df)
            df =  shuffle(df)
            selected = df.loc[df['label'].isin(['white', 'ransomware', 'darknet'])]
            addresses = set()

            # Iterate over the source and target columns and add any unique addresses to the set
            for source, target in zip(data["source"], data["target"]):
                addresses.add(source)
                addresses.add(target)

            selected = selected.loc[selected['address'].isin(addresses)]
            selected = selected[["address", "label"]]
            selected["label"] = selected["label"].map(label_map)
            selected.drop_duplicates(inplace=True)
            adds_of_interest = selected.address.unique()
            selected = dict(selected.values)
            featurizer = OrbitFeaturizer()
            graph = nx.from_pandas_edgelist(data,source="source",target="target",edge_attr="weight",create_using=nx.DiGraph)
            sampled_white =0
            for addr in tqdm(adds_of_interest, total=len(adds_of_interest)):
                if addr not in graph:
                    continue
                lbl = selected[addr]

                if lbl==0 and sampled_white>self.sample:
                    next
                else:
                    if lbl==0:
                        sampled_white = sampled_white + 1

                    #chainlet_gr = self.get_two_chainlet_optimized(addr,graph)
                    chainlet_gr = self.get_two_chainlet(addr, graph)
                    chainlet_gr = nx.convert_node_labels_to_integers(chainlet_gr, first_label=0, ordering='default', label_attribute=None)
                    if lbl != 0:
                        print(lbl)
                    f = featurizer.orbit_featurize(chainlet_gr)

                    data = featurizer.to_pyg_graph()

                    data.y =  torch.tensor([lbl],dtype=torch.int64)

                    data.address = addr
                    if self.test:
                        torch.save(data,
                                   os.path.join(self.processed_dir,
                                                f'data_test_{index}.pt'))
                    else:
                        torch.save(data,
                                   os.path.join(self.processed_dir,
                                                f'data_{index}.pt'))
                    index=index+1
            print("index: ",index)
        self.n = index
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        #return self.data.shape[0]

        return self.num_graphs
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data


