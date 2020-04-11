from abc import ABC

from pjdata.collection import Collection
from pjdata.data import Data
from pjml.tool.abc.transformer import Transformer
from pjml.tool.model import Model


class LightTransformer(Transformer, ABC):
    from pjdata.specialdata import NoData

    def _use_impl(self, data, **kwargs):
        """Each component should implement its core 'apply' functionality."""
        return data

    def apply(self, data: Data = NoData, exit_on_error=True):
        collection_all_nones = data.iscollection and data.all_nones
        if data is None or collection_all_nones or data.failure:
            return Model(self, data, data)

        self._check_nodata(data, self)

        # Disable warnings, measure time and make the party happen.
        self._handle_warnings()  # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        start = self._clock()
        try:
            # Aqui, passa-se _exit_on_error para self de forma que
            # implementadores de conteineres possam acessar o valor
            # dentro de
            # _apply_impl e repassar aos contidos. TODO: Mesmo p/ max_time?
            self._exit_on_error = exit_on_error

            model = self._limit_by_time(
                function=self._apply_impl,
                data=data,
                max_time=self.max_time
            )
            self._check_history(data, model.data, self.transformations('a'))

            # Check result type.
            if not isinstance(model, Model):
                raise Exception(f'{self.name} does not handle {type(model)}!')
        except Exception as e:
            self._handle_exception(e, exit_on_error)
            applied = data.updated(
                self.transformations('a'), failure=str(e)
            )
            model = Model(self, data, applied)
            # TODO: é possível que um container não complete o try acima?
            #  Caso sim, devemos gerar um ContainerModel aqui?

        # TODO: put time_spent inside data (as a "volatile" matrix)?
        time_spent = self._clock() - start
        self._dishandle_warnings()  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # TODO: usar check_history aqui, to guide implementers whenever they
        #  need to implement transformations()

        return model
