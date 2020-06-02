"""Test"""
from pjdata.specialdata import NoData
from pjml.pipeline import Pipeline
from pjml.tool.collection.expand.partition import Partition
from pjml.tool.collection.reduce.reduce import Reduce
from pjml.tool.collection.reduce.summ import RSumm
from pjml.tool.collection.transform.map import Map
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.metric import Metric
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
        Metric(onenhancer=False)
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
        Metric(onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_partition(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(onenhancer=False)),
        RSumm(function='mean', onenhancer=False),
        Reduce(),
        Report('mean ... S: $S', onenhancer=False)
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
        Metric(onenhancer=False),
        Report('metric ... R: $R', onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_with_summ_reduce(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(onenhancer=False)),
        Map(Report('<---------------------- etapa'), onenhancer=False),
        RSumm(function='mean', onenhancer=False),
        Reduce(),
        Report('mean ... S: $S', onenhancer=False)
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_check_architecture(arq='iris.arff'):
    pipe = Pipeline(
        File(arq),
        Partition(partitions=2),
        Map(PCA(), SVMC(), Metric(onenhancer=False)),
        RSumm(field="Y", function='mean', onenhancer=False),
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
        Partition(),
        Map(PCA(), SVMC(), Metric(onenhancer=False)),
        RSumm(field="Y", function='mean', onenhancer=False),
        Report('mean ... S: $S', onenhancer=False)
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
    printable_test()
    test_tsvmc()
    test_split()
    test_metric()
    test_pca()
    test_partition()
    test_split_train_test()
    # test_with_summ_reduce()

    # sanity test
    # test_check_architecture()


if __name__ == '__main__':
    main()
