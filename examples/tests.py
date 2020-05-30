"""Test"""
from pjdata.specialdata import NoData
from pjml.pipeline import Pipeline
from pjml.tool.collection.expand.partition import TPartition
from pjml.tool.collection.reduce.reduce import TRReduce
from pjml.tool.collection.reduce.summ import TRSumm
from pjml.tool.collection.transform.map import TMap
from pjml.tool.data.communication.report import TReport
from pjml.tool.data.evaluation.metric import TMetric
from pjml.tool.data.evaluation.split import Split, SplitTrain, SplitTest
from pjml.tool.data.flow.file import File
from pjml.tool.data.modeling.supervised.classifier.dt import DT
from pjml.tool.data.modeling.supervised.classifier.svmc import SVMC
from pjml.tool.data.processing.feature.reductor.pca import PCA


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
    from pjml.config.operator.many import select
    from pjml.config.operator.reduction.full import full
    from pjml.config.operator.reduction.rnd import rnd
    from pjml.config.operator.single import hold
    from pjml.pipeline import Pipeline
    from pjml.tool.chain import Chain
    from pjml.tool.collection.expand.partition import Partition
    from pjml.tool.collection.reduce.summ import Summ
    from pjml.tool.collection.transform.map import Map
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
    from pjml.tool.data.processing.feature.binarize import Binarize
    from pjml.tool.data.processing.feature.selector.selectkbest import \
        SelectBest
    from pjml.tool.meta.wrap import Wrap
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
    cs = File(arq).cs
    pipe = Pipeline(
        File(arq),
        SVMC()
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_split(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Split(),
        SVMC()
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", str(prior))
    print("Posterior..........\n", str(posterior))


def test_metric(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Split(),
        SVMC(),
        TMetric(onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_pca(arq="iris.arff"):
    cs = File(arq).cs
    pipe = Pipeline(
        File(arq),
        Split(),
        PCA(),
        SVMC(),
        TMetric(onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_partition(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        TPartition(),
        TMap(PCA(), SVMC(), TMetric(onenhancer=False)),
        TRSumm(function='mean', onenhancer=False),
        TReport('mean ... S: $S', onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_split_train_test(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        SplitTrain(),
        SplitTest(),
        PCA(),
        SVMC(),
        TMetric(onenhancer=False),
        TReport('metric ... R: $R', onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_with_summ_reduce(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        TPartition(),
        TMap(PCA(), SVMC(), TMetric(onenhancer=False)),
        TMap(TReport('<---------------------- etapa'), onenhancer=False),
        TRSumm(function='mean', onenhancer=False),
        TRReduce(),
        TReport('mean ... S: $S', onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_check_architecture(arq='iris.arff'):
    pipe = Pipeline(
        File(arq),
        TPartition(partitions=2),
        TMap(PCA(), SVMC(), TMetric(onenhancer=False)),
        TRSumm(function='mean', onenhancer=False),
    )

    # tenho file na frente
    prior_01 = pipe.enhancer.transform(NoData)
    posterior_01 = pipe.model(NoData).transform(NoData)
    prior_02, posterior_02 = pipe.dual_transform(NoData, NoData)

    # Collection uuid depends on data, which depends on consumption.
    for t, *_ in prior_01, prior_02, posterior_01, posterior_02:
        # print(111111111, t.y)
        pass

    assert prior_01.uuid == prior_02.uuid
    assert posterior_01.uuid == posterior_02.uuid


def test_check_architecture2(arq='iris.arff'):
    pipe = Pipeline(
        File(arq),
        TPartition(),
        TMap(PCA(), SVMC(), TMetric(onenhancer=False)),
        TRSumm(function='mean', onenhancer=False),
        TReport('mean ... S: $S', onenhancer=False)
    )

    # tenho file na frente
    prior_ = pipe.enhancer.transform(NoData)
    posterior_ = pipe.model(NoData).transform(NoData)
    posterior_ = pipe.model(NoData).transform((NoData, NoData))
    prior_, posterior_ = pipe.dual_transform(NoData, NoData)
    prior_, posterior_ = pipe.dual_transform(NoData, (NoData, NoData))

    # prior_ = pipe.enhancer().transform()
    # posterior_ = pipe.model().transform()
    # posterior_ = pipe.model().transform()
    # prior_, posterior_ = pipe.dual_transform()
    # prior_, posterior_ = pipe.dual_transform()

    # se não tenho file (tenho split)
    # dado = file()
    # prior = pipe.enhancer().transform(dado)
    # posterior = pipe.model(dado).transform(NoData)

    # prior = pipe.enhancer().transform(dado)
    # posterior = pipe.model(dado).transform()

    # se não tenho split
    # dado = file()
    # train, test = split(dado)
    # prior = pipe.enhancer().transform(train)
    # posterior = pipe.model(train).transform(test)

    # prior_, posterior_ = pipe.dual_transform(train, (train, test))

    # chamando info
    # info = pipe.enhancer().info(train)
    # info = pipe.model(train).info()


def main():
    """Main function"""
    # printable_test()
    # multobj_automl()
    # test_tsvmc()
    test_split()
    # test_metric()
    # test_pca()
    # test_partition()
    # test_split_train_test()
    # test_with_summ_reduce()

    # sanity test
    # test_check_architecture()


if __name__ == '__main__':
    main()
