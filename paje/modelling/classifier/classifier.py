from abc import ABC

from paje.base.component import Component


class Classifier(Component, ABC):
    def apply(self, data):
        self.model = self.model.fit(*data.xy()) # self.model will be set in the child class

    def use(self, data):
        return self.model.predict(data.data_x)
