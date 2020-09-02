import traceback

from cururu.storage import Storage

from pjdata import types as t
from pjdata.mixin.serialization import withSerialization
from pjdata.transformer.enhancer import DSStep
from pjdata.transformer.model import Model
from pjml.config.description.cs.containercs import ContainerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abs.component import Component
from pjml.tool.abs.container1 import Container1
from pjml.tool.abs.mixin.noinfo import withNoInfo


# TODO: remove *args from containers, because they are obsolete when using algebra
class Cache(Container1):
    def __init__(self, *args, storage_alias="default_dump", seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        self.storage = Storage(storage_alias)
        config = {"storage_alias": storage_alias, "seed": seed, "components": components}
        outerself = self

        class Step(withNoInfo, DSStep):
            # TODO: CV() is too cheap to be recovered from storage, specially if
            #  it is a LOO. Maybe transformers could inform whether they are cheap.
            def __init__(self, component: withSerialization, *args):
                super().__init__(outerself.component, *args)
                self.enhancer = outerself.component.enhancer
                self._uuid = outerself.component.enhancer.uuid

            def _transform_impl(self, data: t.Data) -> t.Result:
                hollow = data.hollow(self.enhancer)
                output_data = outerself.storage.fetch(hollow, lock=True)

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
                    outerself.storage.store(output_data.pickable, check_dup=False)

                return output_data

        # TODO: uuid criado pelo cache não funciona para componentes sem Model, checar se overriding aqui está correto
        class Mod(withNoInfo, Model):
            def __init__(self, component: withSerialization, data: t.Data):
                super().__init__(outerself.component, data)
                # TODO: Check if all models can be cheap? We just need its uuid here.
                self.model = outerself.component.model(data)
                self._uuid = outerself.component.model(data).uuid

            def _transform_impl(self, data: t.Data) -> t.Result:
                hollow = data.hollow(self.model)
                output_data = outerself.storage.fetch(hollow, lock=True)

                # Use if still needed  ----------------------------------
                if output_data is None:
                    try:
                        # Do not exit on error, we need to cleanup storage first.
                        output_data = self.model.transform(data, exit_on_error=False)
                    except:
                        outerself.storage.unlock(hollow)
                        traceback.print_exc()
                        exit(0)

                    outerself.storage.store(output_data.pickable, check_dup=False)
                return output_data

        super().__init__(config, Step, Mod, seed, components, enhance, model, deterministic=True)

    def _cfuuid_impl(self, data=None):  # TODO: override uuidimpl as well?
        return self.component.cfuuid(data)

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

# TIP: with algebraic operators, new is not needed anymore
# def __new__(cls, *args, storage_alias="default_dump", seed=0, components=None, **kwargs):
#     """Shortcut to create a ConfigSpace."""
#     if components is None:
#         components = args
#     if all([isinstance(c, Component) for c in components]):
#         return object.__new__(cls)
#     node = Node(params={"storage_alias": FixedP(storage_alias), "seed": FixedP(seed), })
#     return ContainerCS(Cache.name, Cache.path, components, nodes=[node])
