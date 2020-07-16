import traceback
from typing import Dict, Any

from cururu.storage import Storage
from pjdata import types as t
from pjdata.types import Data
from pjml.config.description.cs.containercs import ContainerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abs.container1 import Container1
from pjml.tool.abs.component import Component


class Cache(Container1):
    def _enhancer_info(self, data: t.Data) -> Dict[str, Any]:
        pass

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        pass

    def _flatten(self, transformer, acc=None):
        """Depth-first search to solve nesting of transformers.
        Provides only a rough history, since it does not enter inside unpredictable or complex* components.

        * -> complex components are those - excluding Chain - that generate a sequence of transformations or
        a single transformation different from themselves.

        This code istemporarily here, nut it is useless for Cache, since it resorts direct to UUID."""
        if acc is None:
            acc = []
        if transformer.info.transformers:
            for e in transformer.info.transformers:
                acc = self._flatten(e, acc)
        acc.append(transformer)
        return acc

    def _enhancer_func(self) -> t.Transformation:
        # TODO: CV() is too cheap to be recovered from storage, specially if
        #  it is a LOO. Maybe transformers could inform whether they are cheap.

        def transform(data: Data) -> t.Result:
            enhancer = self.component.enhancer  # TODO: Check if all enhancers are cheap!! We just need its uuid here.
            hollow = data.hollow(enhancer)
            print(111111111111111111111111, data.id)
            output_data = self.storage.fetch(hollow, lock=True)
            print(hollow.id)

            # pra carregar modelo [outdated code here!!]:
            # self.transformer = self.storage.fetch_transformer(
            #     data, self.transformer, lock=True
            # )
            #
            # pra guardar modelo:
            # self.storage.store_transformer(self.transformer, self.fields,
            #                                check_dup=True)

            # Transform if still needed  ----------------------------------
            if output_data is None:
                try:
                    output_data = enhancer.transform(data, exit_on_error=False)
                except:
                    self.storage.unlock(hollow)
                    traceback.print_exc()
                    exit(0)
                # TODO: quando grava um frozen, é preciso marcar isso dealguma forma
                #  para que seja devidamente reconhecido como tal na hora do fetch.
                self.storage.store(output_data, check_dup=False)

            return output_data

        return transform

    def _model_func(self, train: t.Data) -> t.Transformation:
        # TODO: Check if all models can be cheap? We just need its uuid here.
        model = self.component.model(train)

        def transform(test: Data) -> t.Result:
            print(22222222222222222222222222222222222, test.id)
            hollow = test.hollow(model)
            print(hollow.id)
            output_data = self.storage.fetch(hollow, lock=True)

            # Use if still needed  ----------------------------------
            if output_data is None:
                try:
                    # Do not exit on error, we need to cleanup storage first.
                    output_data = model.transform(test, exit_on_error=False)
                    print(output_data.id)
                except:
                    self.storage.unlock(hollow)
                    traceback.print_exc()
                    exit(0)

                self.storage.store(output_data, check_dup=False)
            return output_data

        return transform

    def __new__(cls, *args, storage_alias="default_dump", seed=0, components=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(c, Component) for c in components]):
            return object.__new__(cls)
        node = Node(params={"storage_alias": FixedP(storage_alias), "seed": FixedP(seed),})
        return ContainerCS(Cache.name, Cache.path, components, nodes=[node])

    def __init__(self, *args, storage_alias="default_dump", seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        self.storage = Storage(storage_alias)
        config = {"storage_alias": storage_alias, "seed": seed, "components": components}
        super().__init__(config, seed, components, enhance, model, deterministic=True)
