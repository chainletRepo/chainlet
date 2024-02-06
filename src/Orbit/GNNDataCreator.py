from datetime import datetime, timedelta
from typing import Dict, List, Set
import GraphExtractorBitcoinETL
import NetworkExporter
import BlockReader
from Util import Util
import csv


def main(args: List[str]) -> None:
    block_dir: str = args[1]  # "/media/bcypher/TOSHIBA EXT/chaindata/bitcoin/blocks/"
    edge_dir: str = args[2]  # "/home/bcypher/chaindata/bitcoin/"
    block_times: Dict[int, int] = BlockReader.read_block_information(block_dir)
    print(f"{len(block_times)} blocks exist on the chain.")
    ex: NetworkExporter = NetworkExporter(block_dir)
    extractor: GraphExtractorBitcoinETL = GraphExtractorBitcoinETL()
    # compute for a day
    # most active days
    start_date: datetime = datetime.strptime("01-01-2010", "%m-%d-%Y")
    end_date: datetime = datetime.strptime("01-01-2019", "%m-%d-%Y") + timedelta(days=1)
    dates_between: List[datetime] = Util.get_dates_between(start_date, end_date)
    for date in dates_between:
        day_of_month: int = date.day
        month: int = date.month
        year: int = date.year
        m: str = f"0{month}" if month < 10 else str(month)
        d: str = f"0{day_of_month}" if day_of_month < 10 else str(day_of_month)
        file_name: str = f"graphs/{year}_{m}_{d}graph.csv"
        print(file_name)
        try:
            wr_ad = open(file_name, "w")
            writer = csv.writer(wr_ad)
            blocks: Set[int] = ex.get_blocks(year, month, day_of_month)
            graph = extractor.extract_graph_for(edge_dir, blocks)

            for e in graph.edges():
                writer.writerow(f"{e.get_value()}\t{e.get_value()}\t{e.get_value()}\n")

            wr_ad.close()
            
        except Exception as e:
            print(e)
