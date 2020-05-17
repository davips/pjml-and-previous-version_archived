"""Test"""
from pjml.config.operator.many import select
from pjml.config.operator.reduction.full import full
from pjml.config.operator.reduction.rnd import rnd
from pjml.config.operator.single import hold
from pjml.pipeline import Pipeline, TPipeline
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.partition import Partition
from pjml.tool.collection.reduce.summ import Summ
from pjml.tool.collection.transform.map import Map
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.calc import Calc
from pjml.tool.data.evaluation.mconcat import MConcat
from pjml.tool.data.evaluation.metric import Metric, TMetric
from pjml.tool.data.evaluation.split import Split, TSplit
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File, TFile
from pjml.tool.data.flow.onlyoperation import OnlyApply, OnlyUse
from pjml.tool.data.manipulation.copy import Copy
from pjml.tool.data.modeling.supervised.classifier.dt import DT
from pjml.tool.data.modeling.supervised.classifier.nb import NB
from pjml.tool.data.modeling.supervised.classifier.rf import RF
from pjml.tool.data.modeling.supervised.classifier.svmc import TSVMC
from pjml.tool.data.processing.feature.binarize import Binarize
from pjml.tool.data.processing.feature.reductor.pca import TPCA
from pjml.tool.data.processing.feature.selector.selectkbest import SelectBest
from pjml.tool.meta.wrap import Wrap


def printable_test():
    """toy test."""
    dt_tree = DT()
    dt_tree.disable_pretty_printing()
    print(repr(dt_tree))
    print(dt_tree)
    dt_tree.enable_pretty_printing()
    print()
    print(repr(dt_tree))
    print(dt_tree)


def multobj_automl(arq="abalone3.arff"):
    expr = Pipeline(
        OnlyApply(File(arq), Binarize()),
        Partition(),
        Map(
            Wrap(
                Wrap(
                    select(SelectBest),  # slow??
                    ApplyUsing(select(DT, NB, hold(RF, n_estimators=40))),
                ),
                OnlyApply(Metric(functions=['length'])),
                OnlyUse(Metric(functions=['accuracy', 'error'])),
            )
        ),

        # Report('HISTORY ... S: {history}'),
        Summ(function='mean_std'),

        Report('mean and std ... S: $S'),

        OnlyApply(Copy(from_field="S", to_field="B")),
        OnlyApply(Report('copy S to B ... B: $B')),
        OnlyUse(MConcat(fields=["B", "S"], output_field="S")),
        OnlyUse(Report('comcat B with S (vertical) ... S: $S')),
        OnlyUse(Calc(functions=['flatten'])),
        OnlyUse(Report('flatten S ... S: $S')),
        OnlyUse(Calc(functions=['mean'])),
        OnlyUse(Report('mean S ... S: $S')),

        Report('End ...\n'),
    )

    print('sample .................')
    pipe = full(rnd(expr, n=10), field='S', n=1).sample()

    # TODO: more than one wrap do not work
    best_pipe = pipe.wrapped
    best_pipe = best_pipe.wrapped
    print("The best pipeline is:\n", best_pipe)

    print('apply .................')
    data = Pipeline(File(arq), Binarize()).apply().data
    best_pipe = Chain(pipe.wrapped, Report())
    model = best_pipe.apply(data)
    print('use .................')
    dataout = model.use(data)


def test_tsvmc(arq="iris.arff"):
    cs = TFile(arq).cs
    pipe = TPipeline(
        TFile(arq),
        TSVMC()
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_split(arq="iris.arff"):
    cs = TFile(arq).cs
    pipe = TPipeline(
        TFile(arq),
        TSplit(),
        TSVMC()
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_metric(arq="iris.arff"):
    cs = TFile(arq).cs
    pipe = TPipeline(
        TFile(arq),
        TSplit(),
        TSVMC(),
        TMetric(prior=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_pca(arq="iris.arff"):
    cs = TFile(arq).cs
    pipe = TPipeline(
        TFile(arq),
        TSplit(),
        TPCA(),
        TSVMC(),
        TMetric(prior=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_partition(arq="iris.arff"):
    cs = TFile(arq).cs
    pipe = TPipeline(
        TFile(arq),
        TSplit(),
        TPCA(),
        TSVMC(),
        TMetric(prior=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)

def main():
    """Main function"""
    # printable_test()
    # multobj_automl()
    # test_tsvmc()
    # test_split()
    # test_metric()
    test_pca()


if __name__ == '__main__':
    main()
