import sys
from GraphExtractorBitcoinETL import GraphExtractorBitcoinETL
from OrbitAnalyzer import OrbitAnalyzer
from NetworkExporter import NetworkExporter
import Util
import csv
import BlockReader


class ExperimentDailyOrbits:
    @staticmethod
    def main(args):
        block_dir = args[1]  # "/media/bcypher/TOSHIBA EXT/chaindata/bitcoin/blocks/"
        edge_dir = args[2]  # "/home/bcypher/chaindata/bitcoin/"
        out_dir = args[3]  # "/media/bcypher/TOSHIBA EXT/chaindata/bitcoin/orbits/"
        filter_amount = 0  # Long.parse_long(args[6]);

        block_times = BlockReader.read_block_information(block_dir)
        print(len(block_times) + " blocks exist on the chain.")

        ob = ExperimentDailyOrbits()
        ex = NetworkExporter(block_dir)
        extractor = GraphExtractorBitcoinETL()

        # compute for a day
        for year in range(2018, 2023):
            for month in range(1, 13):
                for day in range(1, 32):
                    m = str(month).zfill(2)
                    d = str(day).zfill(2)
                    file_name = out_dir + str(year) + "_" + m + "_" + d + "address_orbits.csv"
                    print(file_name)
                    
                    file = open(file_name, "w")
                    writer = csv.writer(file)
                    
                    try:
                        blocks = ex.get_blocks(year, month, day)
                        graph = extractor.extract_graph_for(edge_dir, blocks)
                        bf = ob.extract_orbits(graph, filter_amount)
                        writer.writerow(str(bf))
                        
                    except:
                        print(sys.exc_info()[0])
                        
                    finally:
                        file.close()
                        

    def extract_orbits(self, graph, filter_amount):
        bf = []
        bf.append("address")
        
        for i in range(48):  # header
            bf.append("\to" + str(i))
            
        bf.append("\r\n")
        
        oa = OrbitAnalyzer()
        orbits = oa.compute_orbits(graph)
        arr = [0] * 48

        select_nodes = Util.filterby_amount(graph, filter_amount)
        print("Filter amount: " + str(filter_amount) + ", " + str(len(select_nodes)) + " nodes remain from " + str(
            graph.get_num()))

        for n in orbits.keys():
            if n in select_nodes:
                bf.append(n.get_hash_id())
                o_map = orbits.get(n)
                for d in o_map.keys():
                    orbit = o_map.get(d)
                    arr[d] += orbit

                for d in range(48):
                    if d in o_map:
                        bf.append("\t" + str(o_map.get(d)))
                    else:
                        bf.append("\t0")

                bf.append("\r\n")

        return bf
