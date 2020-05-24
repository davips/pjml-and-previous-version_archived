import numpy as np

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.pipeline import Pipeline
from pjml.tool.abc.lighttransformer import LightTransformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.data.communication.report import Report
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File
from pjml.tool.data.modeling.supervised.classifier.nb import NB
from pjml.tool.model.model import Model


class MConcat(LightTransformer, FunctionInspector):
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

    def __init__(self, fields, output_field, direction='horizontal'):
        super().__init__(self._to_config(locals()), deterministic=True)
        self.fields = fields
        self.output_field = output_field

        if direction == 'vertical':
            self.direction = 0
        elif direction == 'horizontal':
            self.direction = 1

        else:
            raise Exception(
                f"Wrong parameter: "
                f"direction = {direction}."
            )

    def _apply_impl(self, data):
        applied = self._use_impl(data, step='a')
        return Model(self, data, applied)

    def _use_impl(self, data, step='u'):
        mats = [data.field(f, self) for f in self.fields]
        dic = {
            self.output_field: np.concatenate(
                tuple(mats), axis=self.direction
            )
        }
        return data.updated(
            self.transformations(step), **dic
        )

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            'input_field1': CatP(choice, items=['S']),
            'input_field2': CatP(choice, items=['S']),
            'direction': CatP(choice, items=[0, 1]),
            'output_field': CatP(choice, items=['S'])
        }
        return TransformerCS(nodes=[Node(params)])

# TODO: create a proper test?
# p = Pipeline(
#     File('iris.arff'),
#     ApplyUsing(NB()),
#     Report('$X $Y $Z'),
#     MConcat(fields=['X','Y','Z'], output_field='A'),
#     Report('$A')
# )
# p.apply()