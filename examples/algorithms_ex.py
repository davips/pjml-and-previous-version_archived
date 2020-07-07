from pjml.config.search.many import select
from pjml.pipeline import Pipeline
from pjml.tool.chain import Chain
from pjml.tool.stream.expand.partition import Partition
from pjml.tool.stream.reduce.summ import Summ
from pjml.tool.stream.transform.map import Map
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.calc import Calc
from pjml.tool.data.evaluation.mconcat import MConcat
from pjml.tool.data.evaluation.metric import Metric
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File
from pjml.tool.data.flow.onlyoperation import OnlyApply, OnlyUse
from pjml.tool.data.manipulation.copy import Copy
from pjml.tool.data.modeling.supervised.classifier.dt import DT
from pjml.tool.data.modeling.supervised.classifier.nb import NB
from pjml.tool.data.modeling.supervised.classifier.rf import RF
from pjml.tool.data.modeling.supervised.classifier.svmc import SVMC
from pjml.tool.data.processing.feature.binarize import Binarize
from pjml.tool.data.processing.feature.scaler.minmax import MinMax
from pjml.tool.data.processing.feature.scaler.std import Std
from pjml.tool.data.processing.feature.selector.selectkbest import SelectBest
from pjml.tool.data.processing.instance.sampler.over.random import OverS
from pjml.tool.data.processing.instance.sampler.under.random import \
    UnderS
from pjml.tool.meta.wrap import Wrap

# print(SelectKB.cs)
# exit()
#
# cs = Pipeline(SelectKB)
# print(cs)
# exit()
#
# s = cs.sample()
# print(s)
# exit()

expr = Pipeline(
    OnlyApply(File("abalone3.arff"), Binarize()),
    Partition(),
    Map(
        Wrap(
            select(SelectBest),
            ApplyUsing(select(DT, RF, NB)),
            OnlyApply(Metric(functions=['length'])),
            OnlyUse(Metric(functions=['accuracy', 'error'])),
            # AfterUse(Metric(function=['diversity']))
        ),
    ),
    Report('HISTORY ... S: {history}'),
    Summ(function='mean_std'),
    Report('mean and std ... S: $S'),

    OnlyApply(Copy(from_field="S", to_field="B")),
    OnlyApply(Report('copy S to B ... B: $B')),
    OnlyUse(MConcat(input_field1="B", input_field2="S",
                    output_field="S", direction='vertical')),
    OnlyUse(Report('comcat B with S (vertical) ... S: $S')),
    OnlyUse(Calc(functions=['flatten'])),
    OnlyUse(Report('flatten S ... S: $S')),
    OnlyUse(Calc(functions=['mean'])),
    OnlyUse(Report('mean S ... S: $S')),

    Report('End ...\n'),

)

# diversidade,
# Lambda(function='$R[0][0] * $R[0][1]', field='r')

print('sample .................')
pipe = full(rnd(expr, n=5), field='S', n=1).sample()

#
# pipes = rnd(expr, n=5)
#
# magia = Multi(pipes) -> Diversity() -> Agrega()
# magia.apply()
# coll = magia.use()
#
# pipe = full(pipes, field='S', n=1).sample()


print('apply .................')
data = Pipeline(File("abalone3.arff"), Binarize()).apply().data

c = Chain(pipe.wrapped, Report())
model = c.apply(data)

print('use .................')
dataout = model.use(data)
