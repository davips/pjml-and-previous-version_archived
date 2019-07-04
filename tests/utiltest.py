from paje.ml.element.element import Element
from paje.base.hps import HPTree
import numpy as np


class SimpElem(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_apply = 0
        self.count_use = 0
        self.count_buid = 0
        self.count_tree = 0
        self.model = None

    def build_impl(self):
        self.count_buid += 1
        print(">>>>> ", self.args_set)
        self.oper = self.args_set['oper']

    def apply_impl(self, data):
        self.count_apply += 1

        value = data.X
        if self.oper == '+':
            value = data.X + data.X
        elif self.oper == '*':
            value = data.X * data.X
        elif self.oper == '/':
            value = data.X / data.X
        elif self.oper == '-':
            value = data.X - data.X
        elif self.oper == '.':
            value = np.dot(data.X, data.X)

        self.model = value.copy()
        data = data.updated(self, X=value)
        return data

    def use_impl(self, data):
        self.count_use += 1
        value = self.model + data.X
        data = data.updated(self, X=value)
        return data

    def tree_impl(self):
        return HPTree({'oper': ['c', ['+', '-', '*', '.']]}, [])


