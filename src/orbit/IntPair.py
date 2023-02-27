class IntPair:
    def __init__(self, x, y):
        self.left = x
        self.right = y

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, IntPair):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        # Use the hash codes of the left and right fields to generate a combined hash code
        left_hash = hash(self.left)
        right_hash = hash(self.right)
        return left_hash ^ right_hash