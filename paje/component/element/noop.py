from paje.component.component import Component
from paje.base.hps import HPTree
from paje.component.element.element import Element


class Noop(Element):
    def build_impl(self):
        self.model = 42 # TODO: better model here?

    def apply_impl(self, data):
        return data

    def use_impl(self, data):
        return data

    def tree_impl(cls, data):
        HPTree({}, [])
