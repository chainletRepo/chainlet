from GraphExtractorBitcoinETL import GraphExtractorBitcoinETL
from datetime import datetime
import networkx as nx
from datetime import datetime, timedelta

class Util:
    
    @staticmethod
    def get_blocks_between_two_date_times(block_times, zone_id, from_time, to_time):
        blocks = set()

        for block_height, epoch_time in block_times.items():
            block_time = datetime.fromtimestamp(epoch_time, tz=zone_id)
            if from_time <= block_time < to_time:
                blocks.add(block_height)

        return blocks

    @staticmethod
    def filter_graph(graph, filter_amount):
        vertices = set(graph.get_vertices())
        for vertex in vertices:
            if vertex.is_transaction():
                delete_this_transaction = True
                for edge in graph.get_out_edges(vertex):
                    value = edge.get_value()
                    if value > filter_amount:
                        delete_this_transaction = False
                        break
                if delete_this_transaction:
                    graph.remove_vertex(vertex)

    @staticmethod
    def extract_graph(edge_dir, blocks):
        extractor = GraphExtractorBitcoinETL()
        graph = extractor.extract_graph_for(edge_dir, blocks)
        return graph

    @staticmethod
    def filter_graph_for_amount(graph, filter_amount):
        vertices = list(graph.nodes())
        new_graph = nx.MultiDiGraph()
        
        for vertex in vertices:
            new_graph.add_node(vertex)
            
        for edge in graph.edges():
            from_vertex = edge.get_from()
            to_vertex = edge.get_to()
            new_graph.add_edge(edge, from_vertex, to_vertex)
            
        for vertex in vertices:
            if vertex.is_transaction():
                delete = True
                for edge in graph.get_out_edges(vertex):
                    if edge.get_value() >= filter_amount:
                        delete = False
                        break
                    
                if delete:
                    new_graph.remove_node(vertex)
                    
        for vertex in vertices:
            if vertex.is_address():
                if len(new_graph.neighbors(vertex)) == 0:
                    new_graph.remove_node(vertex)
                    
        return new_graph

    @staticmethod
    def get_dates_between(start, end):
        start_date = start.strftime(("%d-%m-%Y"))
        end_date = end.strftime(("%d-%m-%Y"))
        dates = set()
        
        while start_date < end_date:
            dates.add(start_date.strftime("%d-%m-%Y"))
            start_date = start_date + timedelta(days=1)
            
        return dates

    @staticmethod
    def create_orbit_header():
        header = "address\tday\tyear\tlabel"
        
        for i in range(48):
            header += f"\to{i}"
            
        return header

    @staticmethod
    def filterby_amount(graph, filter_amount):
        vertices = set()
        
        for vertex in list(graph.nodes()):
            if vertex.is_address():
                value = 0
                for edge in graph.edges(vertex):
                    value += edge.get_value()
                    if value > filter_amount:
                        vertices.add(vertex)
                        break
                    
        return vertices
