"""Test"""
import time
import numpy as np

import pjdata.content.specialdata as s
from pjml.config.description.cs.chaincs import ChainCS
from pjml.config.operator.many import select
from pjml.config.operator.reduction.rnd import rnd
from pjml.config.operator.util import maximize, best, run, lrun, compare, minimize, sort
from pjml.pipeline import Pipeline
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.partition import Partition
from pjml.tool.collection.reduce.reduce import Reduce
from pjml.tool.collection.reduce.summ import Summ
from pjml.tool.collection.transform.map import Map
from pjml.tool.data.communication.cache import Cache
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
    pipe = Pipeline(File(arq), SVMC())
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_split(arq="iris.arff"):
    pipe = Pipeline(File(arq), Split(), SVMC())
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", str(prior))
    print("Posterior..........\n", str(posterior))


def test_metric(arq="iris.arff"):
    pipe = Pipeline(File(arq), Split(), SVMC(), Metric(enhance=False))
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_pca(arq="iris.arff"):
    cs = File(arq).cs
    pipe = Pipeline(File(arq), Split(), PCA(), SVMC(), Metric(enhance=False))
    prior, posterior = pipe.dual_transform()
    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_partition(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("mean ... S: $S", enhance=False),
        Report("$X"),
        Report("$y"),
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
        Metric(enhance=False),
        Report("metric ... R: $R", enhance=False),
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", prior)
    print("Posterior..........\n", posterior)


def test_with_summ_reduce(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Map(Report("<---------------------- etapa"), enhance=False),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("mean ... S: $S", enhance=False),
    )
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", [h.longname for h in prior.history])
    print("Posterior..........\n", [h.longname for h in posterior.history])


def test_cache(arq="iris.arff"):
    pipe = Pipeline(Cache(File(arq), storage_alias="default_sqlite"))
    prior, posterior = pipe.dual_transform()

    print("Prior..............\n", [h.name for h in prior.history])
    print("Posterior..........\n", [h.name for h in posterior.history])


def test_check_architecture(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(partitions=2),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(field="Y", function="mean", enhance=False),
    )

    # tenho file na frente
    prior_01 = pipe.enhancer.transform(s.NoData)
    posterior_01 = pipe.model(s.NoData).transform(s.NoData)
    prior_02, posterior_02 = pipe.dual_transform(s.NoData, s.NoData)

    # Collection uuid depends on data, which depends on consumption.
    for t, *_ in prior_01, prior_02, posterior_01, posterior_02:
        # print(111111111, t.y)
        pass

    assert prior_01.uuid == prior_02.uuid
    assert posterior_01.uuid == posterior_02.uuid


def test_check_architecture2(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(field="Y", function="mean", enhance=False),
        Report("mean ... S: $S", enhance=False),
    )

    # tenho file na frente
    prior_ = pipe.enhancer.transform(s.NoData)
    posterior_ = pipe.model(s.NoData).transform(s.NoData)
    posterior_ = pipe.model(s.NoData).transform((s.NoData, s.NoData))
    prior_, posterior_ = pipe.dual_transform(s.NoData, s.NoData)
    prior_, posterior_ = pipe.dual_transform(s.NoData, (s.NoData, s.NoData))

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


def printing_test(arq="iris.arff"):
    print(Chain(Map(select(File(arq)))))
    exp = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Map(Report("<---------------------- fold"), enhance=False),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("mean ... S: $S", enhance=False),
    )
    print(exp)
    print(select(DT(), SVMC()))

    sel = select(DT(), SVMC())
    print(sel)
    print(Map(DT()))
    exp = ChainCS(
        File(arq),
        Partition(),
        Map(PCA(), select(SVMC(), DT(criterion="gini")), Metric(enhance=False)),
        Report("teste"),
        Map(Report("<---------------------- fold")),
    )
    print(exp)


def random_search(arq="iris.arff"):
    np.random.seed(0)
    exp = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), select(SVMC(), DT(criterion="gini")), Metric(enhance=False)),
        # Map(Report("<---------------------- fold"), enhance=False),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("Mean S: $S", enhance=False),
    )

    expr = rnd(exp, n=10)
    result = best(expr, n=5)
    result.disable_pretty_printing()
    print(result)


def ger_workflow(arq="iris.arff"):
    np.random.seed(0)

    workflow = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), select(SVMC(), DT(criterion="gini")), Metric(enhance=False)),
        Summ(function="mean", enhance=False),
        Reduce(),
        # Report("Mean S: $S", enhance=False),
    )

    return workflow


def util():
    # set state for all seed in pjml
    # we should study if it is necessary a global seed handler to make available multiprocessing.

    def clist():
        np.random.seed(0)
        return rnd(ger_workflow(), n=10)
    # run all the experiment
    print("run all the experiment")
    res1 = run(clist())
    print("----------------------------")

    # run all experiment lazily
    print("run all experiment lazily")
    res2 = lrun(clist())
    print("----------------------------")

    # compare the experiments
    print("compare the two experiment")
    print(compare(res1, res2))
    print("----------------------------")

    # minimize
    print("take the minimun")
    result = minimize(clist(), n=3)
    result.disable_pretty_printing()
    print(result)
    print("----------------------------")

    # maximize
    print("take the maximum")
    result = maximize(clist(), n=3)
    result.disable_pretty_printing()
    print(result)
    print("----------------------------")

    # take the top n
    print("take the top three")
    res1 = best(clist(), n=3)
    res1.disable_pretty_printing()
    print(res1)
    print("----------------------------")

    # # sort
    # print("sort experiment result")
    # res2 = sort(clist())
    # res2.disable_pretty_printing()
    # print(result)
    # print("----------------------------")

    # get a piece
    # print("take the top three")
    # result = best(clist(), n=3)
    # result.disable_pretty_printing()
    # print(result)
    # print("----------------------------")


def main():
    """Main function"""
    # printable_test()
    # test_tsvmc()
    # test_split()
    # test_metric()
    # test_pca()
    # test_partition()
    # test_split_train_test()
    # test_with_summ_reduce()
    # test_cache()
    # printing_test()
    # random_search()
    util()

    # sanity test
    # test_check_architecture()


if __name__ == "__main__":
    main()
