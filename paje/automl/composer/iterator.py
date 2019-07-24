from paje.automl.composer.composer import Composer
from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace


class Iterator(Composer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.reduce = self.materialize(self.config['reduce'])

    def apply_impl(self, data):
        component = self.components[0]
        self.model = []

        data_r = data
        while True:
            aux = component.apply(data)
            if aux is None:
                break
            field = self.reduce.field
            aux = data.updated(self, **{field: aux.get(field)})
            data_r = self.reduce.apply(aux)
            self.model.append(component)
            component = component.next()
            if component is None:
                break

        return data_r

    def use_impl(self, data):
        """ This function will be called by Component in the the 'use()' step.

        Attributes
        ----------
        data: :obj:`Data`
            The `Data` object that represent a dataset used for testing phase.
        """

        data_r = data
        for component in self.model:
            aux = component.use(data)
            field = self.reduce.field
            aux = data.updated(self, **{field: aux.get(field)})
            data_r = self.reduce.use(aux)
            if component.failed:
                raise Exception('Using subcomponent failed! ', component)
        return data_r

    @classmethod
    def tree_impl(cls, config_spaces):
        hps = [
            CatHP('configs', cls.sampling_function,
                  config_spaces=config_spaces[0]),
            CatHP('reduce', cls.sampling_function,
                  config_spaces=config_spaces[1])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @staticmethod
    def sampling_function(config_spaces):
        raise Exception('useless call!!!!!!!!')