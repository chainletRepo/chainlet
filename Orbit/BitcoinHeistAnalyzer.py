import networkx as nx

class BitcoinHeistAnalyzer:
    def __init__(self):
        self.features = {}

    def compute_metrics(self, graph):
        starters = self.find_starters2(graph)
        if starters == None or len(starters) == 0:
            print("There are no starter transactions. Run the starter discovery function first.")
            return None
        
        all_weights = {}
        all_lengths = {}
        all_counts = {}
        s_count = 0

        for starter in starters:
            if graph.has_node(starter):
                weights = {}
                lengths = {}
                counts = {}
                to_visit = []

                for v in graph.successors(starter):
                    to_visit.append(v)
                    weights[v] = 1

                self.walk(graph, weights, lengths, counts, to_visit)

                for an_address in weights.keys():
                    if an_address not in all_weights:
                        all_weights[an_address] = 0
                        all_counts[an_address] = {}

                    if an_address not in all_lengths:
                        all_lengths[an_address] = 10 ** 15

                    all_weights[an_address] = weights[an_address] + all_weights[an_address]

                    if all_lengths[an_address] > lengths[an_address]:
                        all_lengths[an_address] = lengths[an_address]

                    all_node_integer_hash_map = all_counts[an_address]
                    all_node_integer_hash_map[starter] = counts[an_address]

        for addr in all_weights.keys():
            income = 0

            for we in graph.in_edges(addr):
                income += we.get_value()

            node_counts = all_counts[addr]
            loop = 0

            for val in node_counts.values():
                if val > 1:
                    loop = loop + 1

            length = all_lengths[addr]
            weight = all_weights[addr]
            count = node_counts.size()
            neighbor = len(graph.neighbors(addr))
            arr = [count, income, length, loop, neighbor, weight]
            self.features[addr] = arr

        return self.features


    def walk(self, graph, lengths, weights, counts, to_visit):
        current_length = 1
        max_length_so_far = 1
        rounds = 0
        
        while len(to_visit) > 0:
            next_round = set()
            while len(to_visit) > 0:
                of_interest = to_visit.pop(0)

                if of_interest not in counts:
                    counts[of_interest] = 0

                if of_interest not in lengths:
                    lengths[of_interest] = rounds

                if of_interest not in weights:
                    weights[of_interest] = 1

                weight = weights[of_interest]
                counts[of_interest] = 1 + counts[of_interest]

                out_tx = set(graph.successors(of_interest))

                for tx2 in out_tx:
                    id = tx2.get_block_height()
                    if id > current_length:
                        branching = len(set(graph.successors(tx2)))
                        max_length_so_far = id 
                        out_adds = set(graph.successors(tx2))
                        
                        for next_address in out_adds:
                            next_round.add(next_address)

                            if next_address not in weights:
                                weights[next_address] = 0

                            new_weight = weights[next_address] + weight + branching
                            weights[next_address] = new_weight

            current_length = max_length_so_far
            to_visit = to_visit.union(next_round)
            rounds = rounds + 1

    def find_starters_old_implem(self, graph):
        starters = set()
        coinbases = set()
        visited = set()

        for node in list(graph.nodes()):
            if node.is_transaction() and node not in visited:
                visited.add(node)
                in_neigh = list(graph.predecessors(node))
                
                if len(in_neigh) == 0:
                    coinbases.add(node)
                else:
                    is_starter = True
                    in_edges = list(graph.in_edges(node))
                    block_heightof_tx = node.get_block_height()
                    tr_indexof_tx = node.get_tr_index()

                    for we in in_edges:
                        address = we.get_from()
                        for incoming_edge in list(graph.in_edges(address)):
                            block_height = incoming_edge.get_block_height()

                            if block_height < block_heightof_tx:
                                is_starter = False
                                break
                            
                            elif block_height == block_heightof_tx:
                                tr_indexof_incoming = incoming_edge.get_tx_node().get_tr_index()
                                if tr_indexof_incoming < tr_indexof_tx:
                                    is_starter = False
                                    break

                    if is_starter:
                        starters.add(node)

        starters.union(coinbases)
        return starters


    def find_starters2(self, graph):
        starters = set()
        coinbases = set()
        map = {}

        for node in list(graph.nodes()):
            if node.is_transaction:
                in_neigh = list(graph.predecessors(node))
                if len(in_neigh) == 0:
                    coinbases.add(node)
                else:
                    is_starter = True
                    in_neighbors = list(graph.predecessors(node))
                    block_heightof_tx = node.get_block_height()
                    tr_indexof_tx = node.get_tr_index()

                    for from_add in in_neighbors:
                        if from_add in map:
                            min_info = map[from_add]
                            block_height = min_info.get_left()

                            if block_height < block_heightof_tx:
                                is_starter = False
                                break
                            elif block_height == block_heightof_tx:
                                if min_info.get_right() < tr_indexof_tx:
                                    is_starter = False
                                    break

                    if is_starter:
                        starters.add(node)

        starters = starters.union(coinbases)
        return starters


    def find_addr_min_blocks(self, graph):
        starters = {}

        for node in list(graph.nodes()):
            if not node.is_transaction():
                in_neigh = graph.predecessors()
                if len(in_neigh) > 0:
                    min = 10 ** 15
                    tr_indexof_tx = 0

                    for edge in graph.get_in_edges(node):
                        block_heightof_tx = edge.get_block_height()

                        if block_heightof_tx < min:
                            min = block_heightof_tx
                            tr_indexof_tx = edge.get_outp_index()
                        
                    starters[node] = [min, tr_indexof_tx]

        return starters

    
    def get_features(self):
        return self.features


