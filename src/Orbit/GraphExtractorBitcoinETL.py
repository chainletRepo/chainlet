import networkx as nx
import Node
from WeightedEdge import WeightedEdge
import os
import csv

class GraphExtractorBitcoinETL:
    def extract_graph_for(self, dir, blocks):
        graph = nx.MultiDiGraph()
        file_names = set()

        # find out which data files will be opened to find data
        for block_height in blocks:
            b_index = (block_height // 100) * 100
            in_path = f"{dir}/inedges/{b_index}_in.csv"
            out_path = f"{dir}/outedges/{b_index}_out.csv"
            if os.path.isfile(in_path) and os.path.isfile(out_path):
                file_names.add(b_index)

        #open data files, but check if the block height is what we are looking for
        for b_index in file_names:
            in_path = f"{dir}/inedges/{b_index}_in.csv"
            out_path = f"{dir}/outedges/{b_index}_out.csv"

            with open(in_path, 'r') as f_in, open(out_path, 'r') as f_out:
                tx_index = 0

                # read in edges of transactions
                csv_in_reader = csv.reader(f_in)
                for line in csv_in_reader:
                    split = line.strip().split("\t")
                    blk_height = int(split[0])

                    if blk_height not in blocks:
                        continue

                    tx = split[1]
                    tx_node = Node(tx, blk_height, tx_index)
                    graph.add_node(tx_node)

                    for x in range(3, len(split), 2):
                        address = split[x]
                        value = int(split[x + 1])
                        address_node = Node(address)
                        output_index = (x - 3) // 2
                        edge = WeightedEdge(address_node, tx_node, value, output_index, blk_height)
                        graph.add_node(address_node)
                        graph.add_edge(address_node, tx_node, weight=value, edge=edge)

                    tx_index += 1

                tr_index = 0

                # read out edges of transactions
                csv_out_reader = csv.reader(f_out)
                for line in csv_out_reader:
                    split = line.strip().split("\t")
                    blk_height = int(split[0])

                    if blk_height not in blocks:
                        continue

                    tx_hash = split[1]
                    tx_node = Node(tx_hash, blk_height, tr_index)

                    for x in range(3, len(split), 2):
                        address = split[x]
                        output_index = (x - 3) // 2
                        value = int(split[x + 1])
                        address_node = Node(address)
                        edge = WeightedEdge(tx_node, address_node, value, output_index, blk_height)
                        graph.add_node(address_node)
                        graph.add_edge(tx_node, address_node, weight=value, edge=edge)
                        
                    tr_index += 1

        return graph
