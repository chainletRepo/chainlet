from typing import Collection, Dict
import pandas as pd

from Node import Node
from WeightedEdge import WeightedEdge 
import networkx as nx
from Metric import Metric


class RiskMetricAnalyzer:
    COINMIXMINSIZE = 3
    COINMIXING = 2
    SUSPICIOUS = 1
    
    def __init__(self, graph: nx.Multi_di_graph) -> None:
        self.metrics: Dict[Node, Metric] = {}

        tx_types = {}
        tx_add_counts = {}

        # Extract characteristics of tx nodes
        for node in list(graph.nodes()):
            if node.is_transaction():
                in_edges = graph.in_edges(node)
                out_edges = graph.out_edges(node)
                size_in = len(in_edges)
                size_out = len(out_edges)
                tx_add_counts[node] = size_in + size_out
                tx_type = self.coin_mixing_detection(in_edges, size_in, size_out)
                if tx_type != 0:
                    tx_types[node] = tx_type

        # Extract characteristics of address nodes
        for node in list(graph.nodes()):
            if node.is_transaction():
                continue
            
            metric = Metric()

            amount_received = 0
            amount_sent = 0
            first_block = float("inf")
            last_block = float("-inf")
            all_in_edges = list(graph.in_edges(node))
            
            for in_edge in all_in_edges:
                amount_received += in_edge.get_value()
                blk_height = in_edge.get_block_height()
                
                if blk_height > last_block:
                    last_block = blk_height
                    
                if blk_height < first_block:
                    first_block = blk_height

            all_out_edges = list(graph.out_edges(node))
            
            for out_edge in all_out_edges:
                amount_sent += out_edge.get_value()
                blk_height = out_edge.get_block_height()
                
                if blk_height > last_block:
                    last_block = blk_height
                    
                if blk_height < first_block:
                    first_block = blk_height

            metric.set_amount_received(amount_received)
            metric.set_amount_transferred(amount_sent)
            metric.set_block_first_used(first_block)
            metric.set_block_last_used(last_block)

            mixing_transactions = 0
            susp_transactions = 0
            ordinary_transactions = 0
            
            for neigh_tx in list(graph.neighbors(node)):
                if neigh_tx in tx_types:
                    if tx_types[neigh_tx] == RiskMetricAnalyzer.COINMIXING:
                        mixing_transactions += 1
                    elif tx_types[neigh_tx] == RiskMetricAnalyzer.SUSPICIOUS:
                        susp_transactions += 1
                else:
                    ordinary_transactions += 1
                    
                metric.add_to_associated_addresses(tx_add_counts[neigh_tx])

            metric.set_susp_transactions(susp_transactions)
            metric.set_mixing_transactions(mixing_transactions)
            metric.set_ordinary_transactions(ordinary_transactions)

            self.metrics[node] = metric


    def coin_mixing_detection(self, in_edges: list[WeightedEdge], size_in: int, size_out: int) -> int:
        if size_in < RiskMetricAnalyzer.COINMIXMINSIZE or size_out < RiskMetricAnalyzer.COINMIXMINSIZE:
            return 0

        values = []
        for e in in_edges:
            values.append(e.get_value())

        ds = pd.DataFrame(values)
        std = ds.std()
        
        if std < 0.0001 and size_in == size_out:
            return RiskMetricAnalyzer.COINMIXING

        return RiskMetricAnalyzer.SUSPICIOUS
    
    def update_with_new_values(self, new_metrics: Dict[Node, Metric]):
        for node in new_metrics.keys():
            if node in self.metrics:
                old_metric = self.metrics[node]
                new_metric = new_metrics[node]
                
                new_received_amount = old_metric.get_amount_received() + new_metric.get_amount_received()
                old_metric.set_amount_received(new_received_amount)
                
                newSentAmount = old_metric.get_amount_transferred() + new_metric.get_amount_transferred()
                old_metric.set_amount_transferred(newSentAmount)
                
                b1 = new_metric.get_block_first_used()
                b2 = new_metric.get_block_last_used()
                
                if b1 < old_metric.get_block_first_used():
                    old_metric.set_block_first_used(b1)
                    
                if b2 > old_metric.get_block_last_used():
                    old_metric.set_block_last_used(b2)
                    
                old_metric.set_mixing_transactions(old_metric.get_mixing_transactions() + new_metric.get_mixing_transactions())
                old_metric.set_susp_transactions(old_metric.get_susp_transactions() + new_metric.get_susp_transactions())
                self.metrics[node] = old_metric

            else:
                self.metrics[node] = new_metric[node]

    def get_node_metrics(self) -> Dict[Node, Metric]:
        return self.metrics
