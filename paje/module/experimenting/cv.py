from paje.base.component import Component
from paje.base.hps import HPTree


class CV(Component):
    def fields_to_keep_after_use(self):
        return 'all'

    def fields_to_store_after_use(self):
        return 'all'

    def build_impl(self):
        self.model = 42

    def apply_impl(self, data):
        return data

    def use_impl(self, data):
        return data

    def tree_impl(self, data):
        HPTree({}, [])
