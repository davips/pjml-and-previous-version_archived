from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import LightTransformer
from pjml.tool.model import Model


class Copy(LightTransformer, FunctionInspector):
    """Calc to evaluate a given Data field.

    Developer: new metrics can be added just following the pattern '_fun_xxxxx'
    where xxxxx is the name of the new metric.

    Parameters
    ----------
    function
        Name of the function to use to evaluate data objects.
    target
        Name of the matrix with expected values.
    prediction
        Name of the matrix to be evaluated.
    """

    @classmethod
    def _cs_impl(cls):
        raise Exception('Not implemented!')

    def __init__(self, from_field, to_field):
        super().__init__(self._to_config(locals()), deterministic=True)
        self.from_field, self.to_field = from_field, to_field

    def _apply_impl(self, data):
        return Model(self, data, self._use_impl(data, step='a'))

    def _use_impl(self, data, step='u'):
        for field in [self.from_field]:
            if field not in data.matrices:
                raise Exception(
                    f'Impossible to copy: Field '
                    f'{field} does not exist!')

        dic = {
            self.to_field: data.field(self.from_field, self)
        }

        return data.updated(
            self.transformations(step='a'), **dic)
