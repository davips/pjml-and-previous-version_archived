from functools import partial

from time import sleep

from cururu.persistence import Persistence
from cururu.worker import Worker
from pjdata.mixin.printable import disable_global_pretty_printing
from pjml.config.operator.many import select
from pjml.config.operator.reduction.full import full
from pjml.config.operator.reduction.rnd import rnd
from pjml.config.operator.single import hold
from pjml.macro import split
from pjml.pipeline import Pipeline
from pjml.tool.abc.mixin.timers import Timers
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.expand import Expand
from pjml.tool.collection.expand.partition import Partition
from pjml.tool.collection.reduce.summ import Summ
from pjml.tool.collection.transform.map import Map
from pjml.tool.data.communication.cache import Cache
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
import numpy as np

arq = "abalone3.arff"
start = Timers._clock()
disable_global_pretty_printing()
np.random.seed(50)

#
# s = cs.sample()
# print(s)
# exit()
# cache = partial(Cache, engine='dump', blocking=True)

def cache(*args):
    return Cache(
        Cache(
            *args,
            engine='mysql',
            db='paje:XXXXXX@143.107.183.114/paje',
            blocking=not True
        ),
        engine='dump', blocking=True
    )

cache = partial(Cache, engine='sqlite', blocking=False)

# cache = partial(Cache, engine='amnesia', blocking=True)

# expr = Pipeline(File(arq), cache(ApplyUsing(NB())))
# p = expr
# p.apply()
expr = Pipeline(
    OnlyApply(File(arq), cache(Binarize())),
    cache(
        Partition(),
        Map(
            Wrap(
                select(SelectBest),  # slow??
                cache(ApplyUsing(select(DT, NB, hold(RF, n_estimators=40)))),
                OnlyApply(Metric(functions=['length'])),
                OnlyUse(Metric(functions=['accuracy', 'error'])),
                # AfterUse(Metric(function=['diversity']))
            ),
        ),
        # Report('HISTORY ... S: {history}'),
        Summ(function='mean_std'),
    ),
    Report('mean and std ... S: $S'),

    OnlyApply(Copy(from_field="S", to_field="B")),
    OnlyApply(Report('copy S to B ... B: $B')),
    # OnlyUse(Report('>>>>>>  B: {B.shape}')),
    # Report('>>>>>>  S: {S.shape}'),
    OnlyUse(MConcat(fields=["B", "S"], output_field="S")),
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
pipe = full(rnd(expr, n=10), field='S', n=1).sample()

#
# pipes = rnd(expr, n=5)
#
# magia = Multi(pipes) -> Diversity() -> Agrega()
# magia.apply()
# coll = magia.use()
#
# pipe = full(pipes, field='S', n=1).sample()


print('apply .................')
data = Pipeline(File(arq), Binarize()).apply().data

c = Chain(pipe.wrapped, Report())
model = c.apply(data)

print('use .................')
dataout = model.use(data)

print('Tempo: ', '{:.2f}'.format(Timers._clock() - start))
Worker.join()
print('Tempo tot: ', '{:.2f}'.format(Timers._clock() - start))
