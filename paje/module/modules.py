import numpy as np
from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modelling.classifier.ab import AB
from paje.module.modelling.classifier.cb import CB
from paje.module.modelling.classifier.dt import DT
from paje.module.modelling.classifier.knn import KNN
from paje.module.modelling.classifier.mlp import MLP
from paje.module.modelling.classifier.nb import NB
from paje.module.modelling.classifier.rf import RF
from paje.module.modelling.classifier.svm import SVM
from paje.module.preprocessing.supervised.instance.balancer.over. \
    ran_over_sampler import RanOverSampler
from paje.module.preprocessing.supervised.instance.balancer.under. \
    ran_under_sampler import RanUnderSampler
from paje.module.preprocessing.supervised.instance.noise_detector.distance_based.nn import \
    NRNN
from paje.module.preprocessing.unsupervised.feature.scaler.standard \
    import Standard
from paje.module.preprocessing.unsupervised.feature.transformer.drfa import DRFA
from paje.module.preprocessing.unsupervised.feature.transformer.drftag import \
    DRFtAg
from paje.module.preprocessing.unsupervised.feature.transformer.drgrp import \
    DRGRP
from paje.module.preprocessing.unsupervised.feature.transformer.drica import \
    DRICA
from paje.module.preprocessing.unsupervised.feature.transformer.drpca \
    import DRPCA
from paje.module.preprocessing.supervised.feature.selector.statistical.cfs \
    import FilterCFS
from paje.module.preprocessing.supervised.feature.selector.statistical. \
    chi_square import FilterChiSquare
from paje.module.preprocessing.supervised.feature.selector.statistical. \
    f_score import FilterFScore
from paje.module.preprocessing.supervised.feature.selector.statistical. \
    gini_index import FilterGiniIndex
from paje.module.preprocessing.supervised.feature.selector.statistical. \
    t_score import FilterTScore
from paje.module.preprocessing.unsupervised.feature.scaler.equalization \
    import Equalization
from paje.module.preprocessing.unsupervised.feature.transformer.drsrp import \
    DRSRP
from paje.module.preprocessing.unsupervised.feature.transformer.drsvd import \
    DRSVD
from paje.composer.pipeline import Pipeline
from paje.composer.switch import Switch
from paje.base.freeze import Freeze

# TODO: Extract list of all modules automatically from the root package module?
# TODO: add DRFtAg, DRICA when try/catch is implemented in pipeline execution
ready_classifiers = [DT(), KNN(), MLP(), NB(), RF(), SVM()]
# TODO: AB is not ready and CB is too heavy
ready_transformers = [DRPCA(), DRFA(), DRGRP(), DRPCA(), DRSRP()]
ready_scalers = [Equalization(), Standard()]
pip_chi_squared = [Pipeline(components=[
    Freeze(Equalization(), feature_range=(0, 1)),
    FilterChiSquare()
])]
# TODO: FilterCFS() broken:  float division by zero, testar com versão
#  anterior do git. já estava quebrando? ficava escondido pelo try/catch?
ready_filters = [FilterFScore(), FilterGiniIndex(),
                 FilterTScore()] + pip_chi_squared
ready_balancing = [RanOverSampler(), RanUnderSampler()]

knn = KNN()
pca = DRPCA()
std = Standard()
eq = Equalization()
pipe_pca = Pipeline(components=[eq, pca])
pipe2 = Pipeline(components=[pipe_pca, std, pca])
knn2 = Pipeline(components=[pipe2, knn])
mlp = Pipeline(components=[pca, MLP()])

# def_pipelines = [
#     pca,
#     Pipeline(components=[FilterCFS(), mlp])
# ]

default_preprocessors = ready_transformers + ready_scalers + [pca]
default_preprocessors = ready_transformers + ready_scalers + ready_balancing + \
                        ready_filters + [NRNN()]

# switch = Switch(components=[Equalization(), Standard()])
# switch2 = Switch(components=[switch, pipe2, pipe_pca])
# default_preprocessors = [pipe_pca, pipe2, switch, switch2]
# default_preprocessors = [pipe_pca, pca, pipe2]

default_modelers = [knn, knn2, mlp] + ready_classifiers
# default_modelers = [knn]
