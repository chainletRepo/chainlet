from typing import List, Dict, Set, Tuple
from collections import defaultdict
import networkx as nx

from Orbit import Pair


class OrbitAnalyzer:
    CHAINLETDIMENSION = 3

    def __init__(self):
        self.orbits = defaultdict(lambda: defaultdict(int))
        self.pairs = set()
        self.pair_count = 0

    def walk(self, graph: nx.MultiDiGraph, starters: Set) -> Set[Pair]:
        visited, tobe_visited = set(), starters
        while tobe_visited:
            current_tx = next(iter(tobe_visited))
            tobe_visited.remove(current_tx)
            visited.add(current_tx)
            pairsof_this_tx = set()

            block_height = current_tx.get_block_height()
            tr_index = current_tx.get_tr_index()
            for address in graph.successors(current_tx):
                for tx_next in graph.successors(address):
                    if tx_next != current_tx:
                        bh2, tr2 = tx_next.get_block_height(), tx_next.get_tr_index()
                        if bh2 > block_height or (bh2 == block_height and tr2 > tr_index):
                            p = Pair(current_tx, tx_next)
                            pairsof_this_tx.add(p)
                            visited.add(current_tx)
                            if tx_next not in visited:
                                tobe_visited.add(tx_next)

            if not pairsof_this_tx:
                p = Pair(current_tx, current_tx)
                pairsof_this_tx.add(p)

            self.pairs.update(pairsof_this_tx)

        return self.pairs

    def find_induced_graph(self, graph: nx.MultiDiGraph, tx, next_tx) -> nx.MultiDiGraph:
        g2 = nx.DiGraph()
        g2.add_node(tx)
        g2.add_node(next_tx)
        ec = 0

        for ad2 in graph.successors(tx):
            g2.add_node(ad2)
            g2.add_edge(tx, ad2, ec)
            ec += 1

        for ad2 in graph.successors(next_tx):
            g2.add_node(ad2)
            g2.add_edge(next_tx, ad2, ec)
            ec += 1

        for ad2 in graph.predecessors(next_tx):
            g2.add_node(ad2)
            g2.add_edge(ad2, next_tx, ec)
            ec += 1
            
        return g2

    def extract_orbits(self, g3: nx.MultiDiGraph, p: Pair) -> None:
        l, r = p.get_left(), p.get_right()
        first, second, third = set(g3.successors(l)), set(g3.predecessors(r)), set(g3.successors(r))
        common = second.intersection(first)
        f, t, c = len(first), len(third), len(common)
        arr = self.find_type(f, c, t)

        for n1 in first:
            if n1 not in self.orbits:
                self.orbits[n1] = {}
                
            os = self.orbits[n1]
            
            if n1 in common:
                if arr[0] in os:
                    os[arr[0]] += 1
                else:
                    os[arr[0]] = 1
            else:
                if arr[1] in os:
                    os[arr[1]] += 1
                else:
                    os[arr[1]] = 1

        for n1 in third:
            if n1 not in self.orbits:
                self.orbits[n1] = {}
                
            os = self.orbits[n1]
            
            if f == c:
                if arr[1] in os:
                    os[arr[1]] += 1
                else:
                    os[arr[1]] = 1
            elif c < 3:
                if arr[2] in os:
                    os[arr[2]] += 1
                else:
                    os[arr[2]] = 1
            else:
                if arr[1] in os:
                    os[arr[1]] += 1
                else:
                    os[arr[1]] = 1
                    
    def find_type(self, f: int, c: int, t: int):
        if f == 1:
            if t == 1:
                return [3, 4]
            elif t == 2:
                return [5, 6]
            elif t >= self.CHAINLETDIMENSION:
                return [7, 8]
        elif f == 2:
            if c == 1:
                if t == 1:
                    return [9, 10, 11]
                elif t == 2:
                    return [12, 13, 14]
                elif t >= self.CHAINLETDIMENSION:
                    return [15, 16, 17]
            elif c == 2:
                if t == 1:
                    return [18, 19]
                elif t == 2:
                    return [20, 21]
                elif t >= self.CHAINLETDIMENSION:
                    return [22, 23]
        elif f >= self.CHAINLETDIMENSION:
            if c == 1:
                if t == 1:
                    return [24, 25, 26]
                elif t == 2:
                    return [27, 28, 29]
                elif t >= 3:
                    return [30, 31, 32]
            elif c == 2:
                if t == 1:
                    return [33, 34, 35]
                elif t == 2:
                    return [36, 37, 38]
                elif t >= self.CHAINLETDIMENSION:
                    return [39, 40, 41]
            elif c >= self.CHAINLETDIMENSION:
                if t == 1:
                    return [42, 43]
                elif t == 2:
                    return [44, 45]
                elif t >= self.CHAINLETDIMENSION:
                    return [46, 47]
        
        print("Undefined config for "+f+","+c+","+t)
        return None

    def compute_orbits(self, graph: nx.MultiDiGraph):
        analyzer = BitcoinHeistAnalyzer()
        starters = analyzer.find_starters2(graph)
        pairs = self.walk(graph, starters)
        self.pair_count = len(pairs)
        
        for p in pairs:
            if p.get_left() == p.get_right():
                current_tx = p.get_left()
                adds = list(graph.successors(current_tx))
                s = len(adds)
                
                if s > self.CHAINLETDIMENSION:
                    s = self.CHAINLETDIMENSION
                    
                for n in adds:
                    key = s - 1
                    
                    if n not in self.orbits:
                        value = {}
                        value[key] = 1
                        self.orbits[n] = value
                        
                    else:
                        ints = self.orbits[n]
                        
                        if key in ints:
                            ints[key] += 1
                        else:
                            ints[key] = 1
                            
            else:
                g3 = self.find_induced_graph(graph, p.get_left(), p.get_right())
                self.extract_orbits(g3, p)
                
        return self.orbits
    
    def get_orbits(self):
        return self.orbits
    
    def get2Chainlet_count(self):
        return self.pair_count