class Chain:
    def __init__(self, value="", child=None, idx=None):
        self.value = value
        self.child = child

        if idx is None:
            if child is None:
                self.idx = 0
            else:
                self.idx = child.idx + 1
        else:
            self.idx = idx

    def __str__(self):
        aux = str(self.value)
        if self.child is not None:
            aux += str(self.child)
        return aux

    def pop(self):
        aux = []
        chain = self
        idx = chain.idx
        while idx >= 0:
            aux.append(chain.value)
            chain = chain.child
            idx -= 1

        return chain, aux
