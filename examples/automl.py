import numpy

from cururu.persistence import DuplicateEntryException, FailedEntryException
from pjdata.data_creation import read_arff
from pjdata.mixin.printing import disable_global_pretty_printing
from pjml.config.search.many import shuffle, select
from pjml.macro import evaluator
from pjml.pipeline import Pipeline
from pjml.tool.stream.expand.partition import Partition
from pjml.tool.stream.reduce.summ import Summ
from pjml.tool.stream.transform.map import Map
from pjml.tool.data.communication.cache import Cache
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.metric import Metric
from pjml.tool.data.evaluation.split import Split
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File
from pjml.tool.data.flow.onlyoperation import OnlyApply
from pjml.tool.data.flow.source import Source
from pjml.tool.data.flow.sink import Sink
from pjml.tool.data.modeling.supervised.classifier.dt import DT
from pjml.tool.data.modeling.supervised.classifier.nb import NB
from pjml.tool.data.modeling.supervised.classifier.svmc import SVMC
from pjml.tool.data.processing.feature.binarize import Binarize
from pjml.tool.data.processing.feature.scaler.minmax import MinMax
from pjml.tool.data.processing.feature.scaler.std import Std
from pjml.tool.data.processing.instance.sampler.over.random import OverS
from pjml.tool.data.processing.instance.sampler.under.random import \
    UnderS

# Armazenar dataset, sem depender do pacote pjml.
from cururu.pickleserver import PickleServer
from pjml.tool.meta.wrap import Wrap
from pjml.tool.chain import Chain
from pjml.tool.data.manipulation.head import Head
from pjml.tool.data.evaluation.mconcat import MConcat

disable_global_pretty_printing()

#dataout = File('iris.arff').apply().data
#chain = Cache(Chain(Head(), MConcat(input_field1="X", input_field2="Y", output_field="H", direction="vertical")))
#print(chain.apply(dataout).data.H)

#exit()


print('Storing iris...')
try:
    PickleServer().store(read_arff('iris.arff'))
    print('ok!')
except DuplicateEntryException:
    print('Duplicate! Ignored.')

numpy.random.seed(50)
# import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('expr .................')
expr = Pipeline(
    OnlyApply(File('iris.arff')),
    Cache(
    evaluator(
    Wrap(
        shuffle(Std, MinMax),
        # shuffle(Std, select(UnderS, OverS), MinMax),
        ApplyUsing(select(DT, NB)),
    ),
    Metric(functions=['accuracy'])
    )
    )
)



# {history.last.config['function']}
print(expr)
print('sample .................')
pipe = full(rnd(expr, n=10), field='S').sample()
pipe.enable_pretty_printing()
print(f'Pipe:\n{pipe}')
print(f'Wrapped:\n{pipe.unwrap}')
pipe = Chain(File('abalone3.arff'), Binarize(), Split(), pipe.unwrap,
             Metric(), Report())

print('apply .................')
model = pipe.apply()

# print(222222222222222, dataout.history)
# data morre no apply() do predictor


print('use .................')


# print(3333333333333333, dataout.history)
# RUS desaparece no use()

exit(0)

#
# # AutoML ===================================================================
# def evaluator(pipe):
#     # |CS| = |Seq.cs(*)| = |new_cs| = oo
#     return seq(
#         # |CS| = 1
#         Expand,
#
#         # |CS| = 1
#         Multi(sampler(split_type='cv')),
#
#         # |CS| = |Map.cs(pipe)| = |pipe.cs| = oo
#         map(pipe),
#
#         # |CS| = 1
#         Summ(function='mean_std')
#     )
#
#
# # workflow_to_eval = seq(shuffle(PCA, NR, Norm), SVMC)
# workflow_to_eval = seq(shuffle(Mark(PCA), Mutable(NR), Mark(Norm)), SVMC)
#
# workflow = evaluator(seq(workflow_to_eval, Metric(function='accuracy')))
#
# chosen = seq(Multi(Rnd(workflow, size=100)), Map(Report), Best())
# chosen = GA(workflow, data, size=1)
#
# pipeline = Pipeline(
#     chosen.sample(),
#     Report("{history.last.config['function']} $S for dataset {dataset.name}.")
# )
#
# pipe.apply(datain)
# pipe.use(datain)
#
#
# def ga(cs, data, size):
#     return Seq(   # Var()
#         Assign(X, Rnd(cs, size=100)),
#         Loop(
#             Expand(),
#             Multi(X),
#             Assign(Y, Select(n=10, field='s', function='max')),
#             Multi(Y),
#             Crossover(...),
#             Mutation(...),
#             Assign(X, CollectionToTransformers()),
#             Assign(Z, Rnd(cs, size=50)),
#             concat(X,Z)
#         )
#     )
#
#
#
# def rnd(cs, size):
#     return *repeat(cs.sample(), size),
#
#
#
# seq(rnd(seq(ga(workflow, 100), SVMC), 100), Select(n=10, field='s',
# function='max'))
