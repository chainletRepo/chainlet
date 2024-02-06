class Node:
    def __init__(self, tx_hash, block_height=0, tr_index=0):
        self.block_height = block_height
        self.tr_index = tr_index
        self.hash_id = tx_hash
        self.is_transaction = True if tx_hash else False
        self.label = 'unknown'

    def __str__(self):
        return self.hash_id

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.hash_id == other.hash_id
        return False

    def __hash__(self):
        return hash(self.hash_id)

    def is_transaction(self):
        return self.is_transaction

    def is_address(self):
        return not self.is_transaction

    def get_hash_id(self):
        return self.hash_id

    def get_block_height(self):
        if not self.is_transaction:
            print("You should not query id of an address.")
            return None
        return self.block_height

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if self.block_height == other.block_height:
            return 0
        elif self.block_height > other.block_height:
            return 1
        else:
            return -1

    def get_tr_index(self):
        return self.tr_index

    def set_label(self, lbl):
        self.label = lbl

    def get_label(self):
        return self.label
