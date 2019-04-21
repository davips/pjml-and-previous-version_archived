from abc import ABC

from paje.base.component import Component


class Classifier(Component, ABC):
    def apply_impl(self, data):
        self.model.fit(*data.xy())  # self.model will be set in the child class
        data.data_y = self.model.predict(data.data_x)
        return data

    def use_impl(self, data):
        data.data_y = self.model.predict(data.data_x)
        return data
