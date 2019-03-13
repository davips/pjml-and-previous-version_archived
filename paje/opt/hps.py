class HPSpace(object):
    def __init__(self, hps_pre, hps_mod):
        self.hps_prep = hps_pre
        self.actual_hps = hps_mod

    def set_hps(self, hp):
        self.actual_hp = hp


class HPTree(object):
    def __init__(self, data=None, children=None):
        if data is None:
            self.data = {}
        else:
            self.data = data
        if children is None:
            self.children = []
        else:
            self.children = children
