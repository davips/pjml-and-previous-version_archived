from paje.base.component import Component
from paje.base.hps import HPTree


class Noop(Component):
    def build_impl(self):
        self.model = 42

    def apply_impl(self, data):
        return data

    def use_impl(self, data):
        return data

    def tree_impl(cls, data):
        HPTree({}, [])
