from paje.base.component import Component


class Noop(Component):
    def build_impl(self):
        pass

    def apply_impl(self, data):
        return data

    def use_impl(self, data):
        return data