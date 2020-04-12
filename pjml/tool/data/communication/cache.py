import traceback

from cururu.storer import Storer
from pjml.config.description.cs.containercs import ContainerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abc.container1 import Container1
from pjml.tool.abc.transformer import Transformer
from pjml.tool.model.specialmodel import Model, CachedApplyModel


class Cache(Container1, Storer):
    def __new__(cls, *args, fields=None, engine="dump", settings=None,
                blocking=False, seed=0, transformers=None):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Transformer) for t in transformers]):
            return object.__new__(cls)
        node = Node(params={
            'fields': FixedP(fields),
            'engine': FixedP(engine),
            'settings': FixedP(settings)
        })
        return ContainerCS(Cache.name, Cache.path, transformers, nodes=[node])

    def __init__(self, *args, fields=None, engine="dump", settings=None,
                 blocking=False, seed=0, transformers=None):
        if transformers is None:
            transformers = args
        if fields is None:
            fields = ['X', 'Y', 'Z']
        if settings is None:
            settings = {}
        config = self._to_config(locals())
        del config['args']
        super().__init__(config, seed, transformers, deterministic=True)

        self.fields = fields
        self._set_storage(engine, settings, blocking)

    def _apply_impl(self, data):
        # TODO: CV() is too cheap to be recovered from storage, specially if
        #  it is a LOO. Maybe transformers could inform whether they are cheap.

        transformations = self.transformer.transformations('a')
        hollow = data.hollow_extended(transformations=transformations)
        output_data = self.storage.fetch(hollow, self.fields, lock=True)

        # pra carregar modelo [outdated code here!!]:
        # self.transformer = self.storage.fetch_transformer(
        #     data, self.transformer, lock=True
        # )
        #
        # pra guardar modelo:
        # self.storage.store_transformer(self.transformer, self.fields,
        #                                check_dup=True)

        # Apply if still needed  ----------------------------------
        if output_data is None:
            try:
                # model usável
                sub_model = self.transformer.apply(data, exit_on_error=False)
                applied = sub_model.data
            except:
                self.storage.unlock(hollow)
                traceback.print_exc()
                exit(0)

            # TODO: quando grava um frozen, é preciso marcar isso dealguma forma
            #  para que seja devidamente reconhecido como tal na hora do fetch.
            self.storage.store(applied, self.fields, check_dup=False)
        else:
            applied = output_data
            # model não usável
            sub_model = CachedApplyModel(self.transformer, data, applied)

        return Model(self, data, applied, model=sub_model)

    def _use_impl(self, data, model=None, **kwargs):
        training_data = model.data_before_apply
        transformations = self.transformer.transformations('u')
        hollow = data.hollow_extended(transformations=transformations)
        output_data = self.storage.fetch(
            hollow, self.fields,
            training_data_uuid=training_data.uuid, lock=True
        )

        # Use if still needed  ----------------------------------
        if output_data is None:
            # If the apply step was simulated by cache, but the use step is
            # not fetchable, the internal model is not usable. We need to
            # create a usable Model resurrecting the internal transformer or
            # loading it from a previously dumped model.
            if isinstance(model, CachedApplyModel):
                print('It is possible that a previous apply() was successfully'
                      ' stored, but use() with current data wasn\'t.\n'
                      'E.g. you are trying to use with new data, or use() '
                      'was never stored before.\n')
                print('Recovering training data from model to reapply it.'
                      'The goal is to induce a model usable by use()...\n'
                      f'comp: {self.transformer.sid} '
                      f'data: {data.sid}'
                      f'training data: {training_data}')
                # stored_train_data = self.storage.fetch(train_uuid)
                model = self.transformer.apply(training_data)
            try:
                used = model.use(data, exit_on_error=False)
            except:
                self.storage.unlock(hollow,
                                    training_data_uuid=training_data.uuid)
                traceback.print_exc()
                exit(0)
            self.storage.store(used, self.fields,
                               training_data_uuid=training_data.uuid,
                               check_dup=False)
        else:
            used = output_data

        return used

    def transformations(self, step, clean=True):
        """Cache produce no transformations by itself , so it needs to
        override the list of expected transformations."""
        return self.transformer.transformations(step, clean)
