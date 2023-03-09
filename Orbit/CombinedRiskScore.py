from datetime import datetime, timedelta, timezone
from collections import defaultdict
import BlockReader 
from Metric import Metric
from Node import Node
from WeightedEdge import WeightedEdge
import datetime
import pytz
import sys
from RiskMetricAnalyzer import RiskMetricAnalyzer
import networkx as nx



def main(args):
    block_dir = args[1]
    edge_dir = args[2]
    init_year = int(args[3])
    init_month = int(args[4])
    init_day = int(args[5])
    end_year = int(args[6])
    end_month = int(args[7])
    end_day = int(args[8])

    print(f"Scanning : {init_year}/{init_month}/{init_day} to {end_year}/{end_month}/{end_day}")

    block_times = BlockReader.read_block_information(block_dir)
    print(f"{len(block_times)} blocks exist on the chain.")
    zone_id = timezone(timedelta(hours=-6), "America/Chicago")
    zero_hour = datetime.time(datetime(1, 1, 1, 0, 0, 0))

    b_heist_thres_amou = 10000000

    for year in range(init_year, end_year+1):
        print(year)
        end_month_val = 12 if year != end_year else end_month
        for month in range(1, end_month_val+1):
            end_day_val = datetime(year, month+1, 1).day if year == end_year and month == end_month else datetime(year, month, 1).day
            for day in range(1, end_day_val+1):
                graph = make_graph(year, month, day, zero_hour, block_times, zone_id, edge_dir)
                node_metrics = run_analyzers(graph, zone_id, block_times, b_heist_thres_amou)
                heist_metrics = Heist_feature_generator.generate_features(block_times, node_metrics)
                add_heist_metrics_to_node_metrics(node_metrics, heist_metrics)
                convert_metric_times_from_unix_to_zoned_date(zone_id, block_times, node_metrics)


def make_graph(year, month, day, zero_hour, block_times, zone_id, edge_dir):
    graph = nx.Multi_di_graph()
    files = BlockReader.list_files(year, month, day, block_dir)
    
    for file in files:
        edges = BlockReader.read_edge_list(file, zero_hour)
        for e in edges:
            src = Node(e[0], block_times, zone_id)
            dest = Node(e[1], block_times, zone_id)
            wgt = WeightedEdge(e[2])
            graph.add_vertex(src)
            graph.add_vertex(dest)
            graph.add_edge(wgt, src, dest)
            
    return graph


def run_analyzers(graph, zone_id, block_times, b_heist_thres_amou):
    node_metrics = RiskMetricAnalyzer(graph).get_node_metrics()
    Analyzers.add_transaction_counts(graph, node_metrics)
    Analyzers.add_time_horizons(zone_id, block_times, node_metrics)
    Analyzers.add_node_age(zone_id, block_times, node_metrics)
    Analyzers.add_taintedness(b_heist_thres_amou, node_metrics)
    Analyzers.add_random_walks(graph, node_metrics)
    return node_metrics


def add_heist_metrics_to_node_metrics(node_metrics, heist_metrics):
    for node, h in heist_metrics.items():
        try:
            node_metrics[node].add_heist_features(h)
        except KeyError:
            print(f"Heist metric not found for node: {node}")

def convert_metric_times_from_unix_to_zoned_date(zone_id, block_times, node_metrics):
    thousand = 1000
    for node, metric in node_metrics.items():
        block_first_used = metric.get_block_first_used()
        block_last_used = metric.get_block_last_used()
        try:
            epoch_milli = thousand * block_times[block_first_used]
            block_time_first = datetime.datetime.fromtimestamp(epoch_milli / 1000.0, tz=pytz.UTC).astimezone(zone_id)
            metric.set_first_zoned_date_time(block_time_first)
            epoch_milli = thousand * block_times[block_last_used]
            block_time_last = datetime.datetime.fromtimestamp(epoch_milli / 1000.0, tz=pytz.UTC).astimezone(zone_id)
            metric.set_last_zoned_date_time(block_time_last)
        except Exception as e2:
            print(node, metric, str(e2))

def main():
    block_dir = "/media/bcypher/TOSHIBA EXT/chaindata/bitcoin/blocks/"
    edge_dir = "/home/bcypher/chaindata/bitcoin/"
    init_date = datetime.date(2016, 1, 1)
    end_date = datetime.date(2016, 1, 2)
    zone_id = pytz.timezone('America/Chicago')
    zero_hour = datetime.time(0, 0)
    b_heist_thres_amou = 10000000

    block_times = BlockReader.read_block_information(block_dir)
    print(len(block_times), "blocks exist on the chain.")
    node_metrics = defaultdict(Metric)
    for date in daterange(init_date, end_date):
        year, month, day = date.year, date.month, date.day
        print("Processing date:", year, month, day)
        graph = make_graph(year, month, day, zero_hour, block_times, edge_dir)
        node_metrics.update(run_analyzers(graph, zone_id, block_times, b_heist_thres_amou))
    print("Adding heist metrics to node metrics...")
    heist_metrics = Heist_metric_reader.read_heist_information(edge_dir)
    add_heist_metrics_to_node_metrics(node_metrics, heist_metrics)
    print("Converting metric times to zoned date...")
    convert_metric_times_from_unix_to_zoned_date(zone_id, block_times, node_metrics)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

if __name__ == '__main__':
    args = sys.argv
    main(args)