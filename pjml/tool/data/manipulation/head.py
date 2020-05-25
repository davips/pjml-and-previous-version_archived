from pjml.tool.abc.lighttransformer import LightTransformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.model.model import Model



class Head(LightTransformer):
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

    def __init__(self, nrows=5, fields=None):
        if fields is None:
            fields = ['X', 'Y']
        super().__init__({'nrows': nrows}, deterministic=True)
        self.nrows = nrows
        self.fields = fields

    def _apply_impl(self, data):
        applied = self._use_impl(data, step='a')
        return Model(self, data, applied)

    def _use_impl(self, data, step='u'):
        dic = {}
        for field in self.fields:
            dic[field] = data.field(field, self)[0:self.nrows]
        return data.updated(self.transformations(step), **dic)
