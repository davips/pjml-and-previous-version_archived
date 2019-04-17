from abc import ABC

from paje.component import Component


class Classifier(Component, ABC):
    def apply_impl(self, data):
        self.model.fit(*data.xy())  # self.model will be set in the child class

    def use_impl(self, data):
        return self.model.predict(data.data_x)
