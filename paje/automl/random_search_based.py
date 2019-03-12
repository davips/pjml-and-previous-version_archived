from paje.automl.automl import AutoML
from paje.preprocessing.feature_selection.statistical_based.cfs\
        import FilterCFS
from paje.preprocessing.feature_selection.statistical_based.chi_square\
        import FilterChiSquare
from paje.preprocessing.balancer.over.ran_over_sampler\
        import RandomOverSampler
from paje.preprocessing.balancer.under.ran_under_sampler\
        import RandomUnderSampler
from paje.opt.hps import HPTree
from paje.opt.random_search import RandomSearch
import numpy as np
from paje.base.pipeline import Pipeline


class RadomSearchAutoML(AutoML):
    methods = ["all", "white_box", "gray_box", "black_box"]

    def __init__(self, method="all", max_iter=30, fixed=True,
                 deep=5, repetitions=False):
        self.hps_preprocessing = self.hps_modelling = None
        self.prep_comp = self.mode_comp = None
        self.max_iter = max_iter
        self.fixed = fixed
        self.deep = deep
        self.repetitions = repetitions
        self.prep_comp = [FilterCFS, FilterChiSquare,
                          RandomOverSampler, RandomUnderSampler]
        self.mode_comp = [RandomForest]
        self.comps = self.prep_comp + self.mode_comp

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.hps_mode = self.hps_prep = self.hps_comp = []

    def _build_hyperspace(self, data):
        self.hps_prep = [pcomp.hps(data)
                         for pcomp in self.preprocessing_components]
        self.hps_mode = [mcomp.hps(data)
                         for mcomp in self.modelling_components]
        self.comp_hps = self.hps_prep + self.hps_mode

    def __get_random_attr(self, space, conf):
        nro_branches = len(space.children)
        conf.update(space.get_data())

        for k in space.data.keys():
            if space.data[k][0] == 'c':
                conf[k] = np.random.randint(0,
                                            len(space.data[k][1][3]),
                                            1)[0]
            elif space.data[k][0] == 'r':
                conf[k] = space.data[k][0][1] + (
                    np.random.ranf()*space.data[k][0][2])
            elif space.data[k][0] == 'z':
                conf[k] = np.random.randint(space.data[k][0][1],
                                            space.data[k][0][2], 1)

        if nro_branches:
            aux = np.random.randint(nro_branches)
            self.__get_random_attr(space.children[aux], conf)

    def _get_random_attr(self, id_comp):
        confs = []
        for idc in id_comp:
            aux_hps = self.hps_comp[id_comp]
            conf = {}
            self.__get_random_attr(aux_hps, conf)
            confs.append([self.comps[id_comp], conf])

        return conf

    def _next(self):
        # fixed pipeline
        if self.prep_comp:
            if self.fixed:
                # values = np.zeros(len(self.prepo_comp), dtype=int)
                values = []
                sample = range(0, len(self.prep_comp))
                while sample:
                    value = np.random.choice(sample, 1)
                    sample = range(value+1, len(self.prep_comp))
                    values.append(value)
                values = np.array(values, dtype=int)
            # No fixed pipeline
            else:
                n_comp = np.random.randint(self.deep, size=1)
                # with repetitions
                values = np.zeros(n_comp, dtype=int)

                # error check --> if self.deep > len(self.prep_comp)
                # without repetitions
                values = np.random.choice(range(0, self.prep_comp),
                                          size=n_comp,
                                          replace=self.repetitions)

            # Select preprocessing components
            sel_comp = values

        # Select modelling components
        if self.mode_comp:
            sel_mode_comp = np.random.choice(range(0, self.mode_comp),
                                             size=1)
            sel_comp = sel_comp + sel_mode_comp

        conf = self._get_random_attr(sel_comp)

        return conf

    def _objective(self, conf, data):
        pipe = Pipeline(conf)
        perf = self.evaluator.eval(pipe)

        return np.mean(perf)

    def _fmin(self, data):
        best_conf = self._next()
        best_value = self._objective(best_conf, data)

        for t in range(1, self.max_iter):
            conf = self._next()
            value = self._objective(conf, data)

            if value < best_value:
                best_value = value
                best_conf = conf

        return best_value, best_conf

    def apply(self, data):
        self._build_hyperspace(data)
        self.evaluator = Evaluator(data, "cv", 5, "auc")
        best_value, best_conf = self._fmin(data)
        self.best_pipeline = Pipeline(best_conf).apply(data)

    def use(self, data):
        self.best_pipeline.use(data)


class Evaluator():
    def __init__(self, data, type_break="cv", steps=10, metric):
        
    def eval(self, pipe):
        pass
