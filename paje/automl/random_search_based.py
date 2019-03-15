from paje.automl.automl import AutoML
from paje.preprocessing.feature_selection.statistical_based.cfs\
        import FilterCFS
from paje.preprocessing.feature_selection.statistical_based.chi_square\
        import FilterChiSquare
from paje.preprocessing.balancer.over.ran_over_sampler\
        import RanOverSampler
from paje.preprocessing.balancer.under.ran_under_sampler\
        import RanUnderSampler
from paje.base.hps import HPTree
import numpy as np
from paje.pipeline.pipeline import Pipeline
from paje.data.data import Data
from paje.modelling.random_forest import RandomForest
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics


class RadomSearchAutoML(AutoML):
    methods = ["all", "white_box", "gray_box", "black_box"]

    def __init__(self, method="all", max_iter=30, fixed=True,
                 deep=5, repetitions=False, random_state=0):
        self.hps_preprocessing = self.hps_modelling = None
        self.prep_comp = self.mode_comp = None
        self.max_iter = max_iter
        self.fixed = fixed
        self.deep = deep
        self.repetitions = repetitions
        self.prep_comp = [FilterCFS, FilterChiSquare,
                          RanOverSampler, RanUnderSampler]
        self.mode_comp = [RandomForest]
        self.comps = self.prep_comp + self.mode_comp
        self.random_state = random_state

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.hps_mode = []
        self.hps_prep = []
        self.hps_comp = []
        self.best_pipeline = None

    def _build_hyperspace(self, data):
        self.hps_prep = [pcomp.hps(data)
                         for pcomp in self.prep_comp]
        self.hps_mode = [mcomp.hps(data)
                         for mcomp in self.mode_comp]
        self.comp_hps = self.hps_prep + self.hps_mode

    def __get_random_attr(self, space, conf):
        nro_branches = len(space.children)
        # conf.update(space.get_data())
        for k in space.data.keys():
            # print(space.data[k])
            if space.data[k][0] == 'c':
                aux = np.random.randint(0, len(space.data[k][1]), 1)[0]
                conf[k] = space.data[k][1][aux]

            elif space.data[k][0] == 'r':
                conf[k] = ((space.data[k][2]-space.data[k][1])*np.random.ranf()) + space.data[k][1]
            elif space.data[k][0] == 'z':
                conf[k] = np.random.randint(space.data[k][1],
                                            space.data[k][2], 1)[0]

        if nro_branches:
            aux = np.random.randint(nro_branches)
            self.__get_random_attr(space.children[aux], conf)

    def _get_random_attr(self, id_comp, hps_comps, comps):
        confs = []
        # print(id_comp)
        # print(hps_comps)
        # print(comps)
        for idc in id_comp:
            aux_hps = hps_comps[idc]
            conf = {}
            self.__get_random_attr(aux_hps, conf)
            confs.append([comps[idc], conf])

        return confs

    def _next(self):
        # fixed pipeline

        conf_prep = []
        if self.prep_comp:
            if self.fixed:
                # values = np.zeros(len(self.prepo_comp), dtype=int)
                values = []
                sample = range(0, len(self.prep_comp))
                while sample:
                    value = np.random.choice(sample, 1)[0]
                    sample = range(value+1, len(self.prep_comp))
                    values.append(value)
                values = np.array(values, dtype=int)
            # No fixed pipeline
            else:
                # print("No fixed")
                n_comp = np.random.randint(self.deep, size=1)[0]
                # with repetitions
                values = np.zeros(n_comp, dtype=int)

                # error check --> if self.deep > len(self.prep_comp)
                # without repetitions
                values = np.random.choice(range(0, len(self.prep_comp)),
                                          size=n_comp,
                                          replace=self.repetitions)

            # Select preprocessing components
            # print(values.tolist())
            conf_prep = self._get_random_attr(values.tolist(), self.hps_prep,
                                              self.prep_comp)

        # Select modelling components
        conf_mode = []
        if self.mode_comp:
            values = np.random.choice(range(0, len(self.mode_comp)),
                                      size=1)
            # print(values.tolist())
            conf_mode = self._get_random_attr(values.tolist(), self.hps_mode,
                                              self.mode_comp)

        # print(conf_prep)
        # print(conf_mode)
        confs = conf_prep + conf_mode
        # print(confs)
        return confs

    def _objective(self, conf, data):
        pipe = Pipeline(conf)
        perf = self.evaluator.eval(pipe, data)
        # print(np.mean(perf))

        return np.mean(perf)

    def _fmin(self, data):
        best_conf = self._next()
        best_value = self._objective(best_conf, data)

        print(best_value)
        for t in range(1, self.max_iter):
            conf = self._next()
            value = self._objective(conf, data)

            if value < best_value:
                best_value = value
                print(best_value)
                best_conf = conf

        return best_value, best_conf

    def apply(self, data):
        self._build_hyperspace(data)
        self.evaluator = Evaluator(data, Metrics.error, "cv",
                                   3, self.random_state)
        best_value, best_conf = self._fmin(data)
        print(best_conf)
        print(best_value)
        self.best_pipeline = Pipeline(best_conf)
        self.best_pipeline.apply(data)

    def use(self, data):
        return self.best_pipeline.use(data)

    def hps(self, data):
        pass
