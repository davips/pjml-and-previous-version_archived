import numpy as np
from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
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
from paje.module.preprocessing.unsupervised.feature.scaler.standard \
    import Standard
from paje.module.preprocessing.unsupervised.feature.transformer.drfa import DRFA
from paje.module.preprocessing.unsupervised.feature.transformer.drftag import DRFtAg
from paje.module.preprocessing.unsupervised.feature.transformer.drgrp import DRGRP
from paje.module.preprocessing.unsupervised.feature.transformer.drica import DRICA
from paje.module.preprocessing.unsupervised.feature.transformer.drpca \
    import DRPCA
from paje.module.preprocessing.supervised.feature.selector.statistical.cfs \
    import FilterCFS
from paje.module.preprocessing.unsupervised.feature.scaler.equalization \
    import Equalization
from paje.module.preprocessing.unsupervised.feature.transformer.drsrp import DRSRP
from paje.module.preprocessing.unsupervised.feature.transformer.drsvd import DRSVD
from paje.pipeline.pipeline import Pipeline

# TODO: Extract list of all modules automatically from the package module.
ready_transformers = [DRPCA, DRFA,  DRGRP,  DRPCA, DRSRP]  #TODO: add DRFtAg, DRICA when try/catch is implemented in pipeline execution
ready_scalers = [Equalization, Standard]
ready_classifiers = [RF, MLP]

default_preprocessors = ready_transformers + ready_scalers
default_modelers = ready_classifiers