class WeightedEdge:
    def __init__(self, frm, to, value, output_index, block_height):
        self.from_node = frm
        self.to_node = to
        self.value = value
        self.output_index = output_index
        self.block_height = block_height

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, WeightedEdge):
            return False
        return self.from_node == other.from_node and self.to_node == other.to_node and \
               self.value == other.value and self.output_index == other.output_index

    def __hash__(self):
        result = self.from_node.__hash__()
        result = 31 * result + int(self.value ^ (self.value >> 32))
        result = 31 * result + self.to_node.__hash__()
        result = 31 * result + self.output_index
        return result

    def get_from_node(self):
        return self.from_node

    def get_to_node(self):
        return self.to_node

    def get_value(self):
        return self.value

    def get_tx_node(self):
        if self.from_node.is_transaction():
            return self.from_node
        else:
            return self.to_node

    def __str__(self):
        return f"Weighted_edge{{from={self.from_node}, to={self.to_node}}}"

    def get_output_index(self):
        return self.output_index

    def get_block_height(self):
        return self.block_height