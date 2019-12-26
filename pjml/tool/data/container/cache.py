from cururu.amnesia import Amnesia
from cururu.pickleserver import PickleServer
from pjml.config.configspace import ConfigSpace
from pjml.config.distributions import choice
from pjml.config.parameter import CatP
from pjml.tool.base.transformer import Transformer


def cache(component, engine="file", settings=None):
    """Shortcut to create a ConfigSpace for Cache.

    Parameters
    ----------
    component

    engine
    settings

    Returns
    -------

    """
    return Cache.cs(component, engine=engine, settings=settings)


class Cache(Transformer):
    def __init__(self, transformer, engine="file", settings=None):
        super().__init__(self._to_config(locals()), engine)

        if settings is None:
            settings = {'db': '/tmp'}
        self.uuid = self.transformer.uuid
        if engine == "amnesia":
            from paje.storage.mysql import MySQL
            self._storage = Amnesia(**settings)
        elif engine == "mysql":
            from paje.storage.mysql import MySQL
            self._storage = MySQL(**settings)
        elif engine == "sqlite":
            from paje.storage.sqlite import SQLite
            self._storage = SQLite(**settings)
        elif engine == "nested":
            from paje.storage.mysql import MySQL
            from paje.storage.sqlite import SQLite
            self._storage = MySQL(db=settings['db'], nested=SQLite())
        elif engine == "file":
            self._storage = PickleServer(**settings)
        else:
            raise Exception('Unknown engine:', engine)

        self._storage._dump = 'dump' in config  # TODO: not used yet!

    @staticmethod
    def sampling_function(config_spaces):
        return [choice(config_spaces).sample()]

    def apply_impl(self, data):
        # TODO: CV() is too cheap to be recovered from storage,
        #  specially if it is a LOO.
        #  Maybe some components could inform whether they are cheap.
        output_data, started = self._storage.get_result(
            self.transformer, self.op, self.train_data_uuid__mutable(), data
        )

        if started:
            self.component._train_data_uuid__mutable = \
                self.train_data_uuid__mutable()
            if self.component.failed:
                print(f"Won't apply on data {data.name}"
                      f"\nCurrent {self.component.name} already failed before.")

            if self.component.locked_by_others:
                print(f"Won't apply {self.component.name} on data "
                      f"{data.name}\n"
                      f"Currently probably working at node "
                      f"[{self.component.host}].")
            return output_data

        # Apply if still needed  ----------------------------------
        self._storage.lock(self.component, data)
        output_data = self.component.apply(data)
        self._storage.store_result(self.component, data, output_data)
        return output_data

    def use_impl(self, data):
        output_data, started = self._storage.get_result(
            self.component, self.op, self.train_data_uuid__mutable(), data
        )

        if started:
            if self.component.locked_by_others:
                print(f"Won't use {self.component.name} on data "
                      f"{data.name}\n"
                      f"Currently probably working at {self.component.host}.")

            if self.component.failed:
                print(f"Won't use on data {data.sid()}\n"
                      f"Current {self.component.name} already failed before.")
            return output_data

        # Use if still needed  ----------------------------------
        self._storage.lock(self.component, data)

        # If the component was applied (probably simulated by storage),
        # but there is no model, we reapply it...
        if self.component.model is None:
            print('It is possible that a previous apply() was '
                  'successfully stored, but its use() wasn\'t.'
                  'Or you are trying to use in new data.')
            print(
                'Trying to recover training data from storage to apply '
                'just to induce a model usable by use()...\n'
                f'comp: {self.component.sid()}  data: {data.sid()} ...')
            train_uuid = self.train_data_uuid__mutable()
            stored_train_data = self._storage.get_data_by_uuid(train_uuid)
            self.component.apply_impl(stored_train_data)

        output_data = self.component.use(data)
        self._storage.store_result(self.component, data, output_data)
        return output_data

    @classmethod
    def _cs_impl(cls):
        params = {'engine': CatP(
            choice, items=["amnesia","mysql","sqlite","nested"]
        )} # TODO: cada engine é um ramo!
        return ConfigSpace(params=params)
