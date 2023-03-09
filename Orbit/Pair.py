class Pair:
    def __init__(self, left, right):
        assert left is not None
        assert right is not None
        self.left = left
        self.right = right

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def __hash__(self):
        # Combine the hash codes of the left and right fields using XOR
        return hash(self.left) ^ hash(self.right)

    def __eq__(self, other):
        if not isinstance(other, Pair):
            return False
        return self.left == other.get_left() and self.right == other.get_right()