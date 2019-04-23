from typing import Dict, List


class HPTree(object):
    def __init__(self, dic, children):
        self.dic = dic
        self.children = children

    def expand(self) -> (Dict, List):
        return self.dic, self.children