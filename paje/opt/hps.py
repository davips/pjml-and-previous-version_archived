class HPSpace(object):
    def __init__(self, hptree):
        self.hps = hptree
        self.actual_hps = None

    def set_hps(self, hp):
        self.actual_hp = hp

class HPTree(object):
    def __init__(self, data=None, children=None):
        if data is not None:
            self.data = {}
        else:
            self.data = data
        if children is not None:
            self.children = []
        else:
            self.children = children
