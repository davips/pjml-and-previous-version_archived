import traceback
from typing import Dict, Any

from cururu.storage import Storage
from pjdata import types as t
from pjdata.mixin.serialization import withSerialization
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.transformer import Transformer
from pjdata.types import Data
from pjml.config.description.cs.containercs import ContainerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abs.container1 import Container1
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Cache(Container1):
    # def _flatten(self, transformer, acc=None):
    #     """Depth-first search to solve nesting of transformers.
    #     Provides only a rough history, since it does not enter inside unpredictable or complex* components.
    #
    #     * -> complex components are those - excluding Chain - that generate a sequence of transformations or
    #     a single transformation different from themselves.
    #
    #     This code istemporarily here, nut it is useless for Cache, since it resorts direct to UUID."""
    #     if acc is None:
    #         acc = []
    #     if transformer.info.transformers:
    #         for e in transformer.info.transformers:
    #             acc = self._flatten(e, acc)
    #     acc.append(transformer)
    #     return acc

    def __new__(cls, *args, storage_alias="default_dump", seed=0, components=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(c, Component) for c in components]):
            return object.__new__(cls)
        node = Node(params={"storage_alias": FixedP(storage_alias), "seed": FixedP(seed), })
        return ContainerCS(Cache.name, Cache.path, components, nodes=[node])

    def __init__(self, *args, storage_alias="default_dump", seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        self.storage = Storage(storage_alias)
        config = {"storage_alias": storage_alias, "seed": seed, "components": components}
        outerself = self

        class Enh(withNoInfo, Enhancer):
            # TODO: CV() is too cheap to be recovered from storage, specially if
            #  it is a LOO. Maybe transformers could inform whether they are cheap.
            def __init__(self, component: withSerialization, *args):
                super().__init__(outerself.component, *args)
                self.enhancer = outerself.component.enhancer

            def _transform_impl(self, data: t.Data) -> t.Result:
                hollow = data.hollow(self.enhancer)
                print(111111111111111111111111, data.id)
                output_data = outerself.storage.fetch(hollow, lock=True)
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
                        output_data = self.enhancer.transform(data, exit_on_error=False)
                    except:
                        outerself.storage.unlock(hollow)
                        traceback.print_exc()
                        exit(0)
                    # TODO: quando grava um frozen, é preciso marcar isso dealguma forma
                    #  para que seja devidamente reconhecido como tal na hora do fetch.
                    outerself.storage.store(output_data, check_dup=False)

                return output_data

        class Mod(withNoInfo, Model):
            def __init__(self, component: withSerialization, data:t.Data):
                super().__init__(outerself.component, data)
                # TODO: Check if all models can be cheap? We just need its uuid here.
                self.model = outerself.component.model(data)

            def _transform_impl(self, data: t.Data) -> t.Result:
                print(22222222222222222222222222222222222, data.id)
                hollow = data.hollow(self.model)
                print(hollow.id)
                output_data = outerself.storage.fetch(hollow, lock=True)

                # Use if still needed  ----------------------------------
                if output_data is None:
                    try:
                        # Do not exit on error, we need to cleanup storage first.
                        output_data = self.model.transform(data, exit_on_error=False)
                        print(output_data.id)
                    except:
                        outerself.storage.unlock(hollow)
                        traceback.print_exc()
                        exit(0)

                    self.storage.store(output_data, check_dup=False)
                return output_data

        super().__init__(config, seed, components, Enh, Mod, enhance, model, deterministic=True)

    def _cfuuid_impl(self, data=None):  # TODO: override uuidimpl as well?
        return self.component.cfuuid(data)