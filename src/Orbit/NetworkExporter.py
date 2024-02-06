from datetime import datetime, timedelta
import pytz
from typing import Dict, Set, Tuple
import BlockReader
from GraphExtractorBitcoinETL import GraphExtractorBitcoinETL


class NetworkExporter:
    block_times: Dict[int, int]
    our_time_zone = pytz.timezone('America/Chicago')

    def __init__(self, block_dir: str) -> None:
        try:
            self.block_times = BlockReader.read_block_information(block_dir)
        except Exception as e:
            raise RuntimeError(e)

    @classmethod
    def from_args(cls, args: Tuple[str, str, str, str, str]) -> "NetworkExporter":
        block_dir, edge_dir, year, month, day = args
        return cls(block_dir).upload_day_tx_network(edge_dir, int(year), int(month), int(day))

    def upload_day_tx_network(self, edge_dir: str, year: int, month: int, day: int) -> None:
        blocks = self.get_blocks(year, month, day)

        if len(blocks) == 0:
            print(f"{day}/{month}/{year} data is missing.")
            return

        extractor = GraphExtractorBitcoinETL()
        graph = extractor.extract_graph_for(edge_dir, blocks)
        vertex_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        print(f"{day}/{month}/{year}: {vertex_count} vertices and {edge_count} edges")

        if edge_count > 0:
            pass
            
    def get_blocks(self, year: int, month: int, day: int) -> Set[int]:
        from_time = datetime(year, month, day, tzinfo=self.our_time_zone)
        to_time = from_time + timedelta(days=1)
        blocks = set()

        for block_height, epoch_time in self.block_times.items():
            block_time = datetime.fromtimestamp(epoch_time, tz=self.our_time_zone)
            if from_time <= block_time < to_time:
                blocks.add(block_height)

        return blocks
