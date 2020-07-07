from pjdata.mixin.printing import withPrinting, disable_global_pretty_printing, \
    enable_global_pretty_printing
from pjml.pipeline import Pipeline
from pjml.tool.chain import Chain
from pjml.tool.stream.expand.partition import Partition
from pjml.tool.stream.reduce.summ import Summ
from pjml.tool.stream.transform.map import Map
from pjml.tool.data.communication.cache import Cache
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.calc import Calc
from pjml.tool.data.evaluation.mconcat import MConcat
from pjml.tool.data.evaluation.metric import Metric
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File
from pjml.tool.data.flow.onlyoperation import OnlyApply, OnlyUse
from pjml.tool.data.flow.sink import Sink
from pjml.tool.data.manipulation.copy import Copy
from pjml.tool.data.modeling.supervised.classifier.rf import RF
from pjml.tool.data.processing.feature.binarize import Binarize
from pjml.tool.data.processing.feature.scaler.minmax import MinMax
from pjml.tool.data.processing.instance.sampler.over.random import OverS
from pjml.tool.data.processing.instance.sampler.under.random import UnderS
from pjml.tool.meta.wrap import Wrap

disable_global_pretty_printing()
d = File("abalone3.arff").apply().data

print('Construindo...')
# pipe = Pipeline(
#     OnlyApply(File("abalone3.arff")),
#     Cache(Binarize()),
#     Partition(),
#     Map(
#         Wrap(
#             MinMax(),
#             Cache(ApplyUsing(RF())),
#             OnlyApply(Metric(functions=['length'])),
#             OnlyUse(Metric(functions=['accuracy', 'error'])),
#             # AfterUse(Metric(function=['diversity']))
#         ),
#     ),
#     Summ(function='mean_std'),
#     Report('$S'),
# )

pipe = Pipeline(
    File("abalone3.arff"),
    Binarize(),
    Partition(),
    Map(
        UnderS(sampling_strategy='not minority'),
        # RF(),
        Cache(RF()),
        Metric()
    ),
    Summ(function='mean_std'),
    Report('mean S --> \n$S'),

    Report('mean S --> $S'),
    OnlyApply(Copy(from_field="S", to_field="B")),
    OnlyUse(
        MConcat(fields=["S", "S"], output_field="S", direction='vertical')),
    Calc(functions=['flatten']),
    Report('mean S --> $S')
)

print('Applying...')
model = pipe.apply()
if model.data:
    for i, t in enumerate(model.data.history):
        print(f'hist {i}', t)
# exit()

print('Using...')
enable_global_pretty_printing()
d2 = model.use()
