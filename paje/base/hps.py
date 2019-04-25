from typing import Dict, List


class HPTree(object):
    def __init__(self, dic, children):
        self.dic = dic
        self.children = children

    def expand(self) -> (Dict, List):
        return self.dic, self.children

    def __str__(self, depth=''):
        rows = [depth + str(self.dic) + '\n']
        depth += '    '
        for child in self.children:
            rows.append(child.__str__(depth))
        return ''.join(rows)

    __repr__ = __str__
