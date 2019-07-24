import numpy as np
from paje.automl.automl import AutoML
from paje.automl.composer.iterator import Iterator
from paje.automl.composer.pipeline import Pipeline
from paje.evaluator.evaluator import EvaluatorClassif
from paje.ml.element.posprocessing.metric import Metric
from paje.ml.element.posprocessing.reduce import Reduce
from paje.ml.element.posprocessing.summ import Summ
from paje.ml.element.preprocessing.supervised.instance.sampler.cv import CV


class RandomAutoML(AutoML):

    def __init__(self,
                 preprocessors,
                 modelers,
                 pipe_length,
                 repetitions,
                 random_state,
                 storage_settings_for_components=None,
                 **kwargs):
        """
        AutoML
        :param preprocessors: list of modules for balancing,
            noise removal, sampling etc.
        :param modelers: list of modules for prediction
            (classification or regression etc.)
        :param repetitions: how many times can a module appear
            in a pipeline
        :param method: TODO
        :param max_iter: maximum number of pipelines to evaluate
        :param max_depth: maximum length of a pipeline
        :param static: are the pipelines generated always exactly
            as given by the ordered list preprocessors + modelers?
        :param fixed: are the pipelines generated always with
            length max(max_depth, len(preprocessors + modelers))?
        :param random_state: TODO
        :return:
        """

        AutoML.__init__(self,
                        components=preprocessors + modelers,
                        evaluator=EvaluatorClassif(),
                        **kwargs)

        # These attributes identify uniquely AutoML.
        # This structure is necessary because the AutoML is a Component and it
        # could be used into other Components, like the Pipeline one.
        # build_impl()
        self.repetitions = repetitions
        self.pipe_length = pipe_length
        # __init__()
        self.modelers = modelers
        self.preprocessors = preprocessors

        if not isinstance(modelers, list) or \
                not isinstance(preprocessors, list):
            print(modelers)
            print(preprocessors)
            raise TypeError("The modelers/preprocessors must be list.")

        if not modelers:
            raise ValueError("The list length must be greater than one.")

        # Other class attributes.
        # These attributes can be set here or in the build_impl method. They
        # should not influence the AutoML final result.
        self.storage_settings_for_components = storage_settings_for_components

        # Class internal attributes
        # Attributes that were not parameterizable
        self.best_eval = float('-Inf')
        self.best_pipe = None
        self.curr_eval = None
        self.curr_pipe = None

        self.random_state = random_state
        np.random.seed(self.random_state)

    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        components = self.choose_modules()
        tree = Pipeline.tree(config_spaces=components)

        config = tree.sample()

        config['random_state'] = self.random_state
        # self.curr_pipe = Pipeline(
        #     config, storage_settings=self.storage_settings_for_components
        # )
        self.curr_pipe = config
        return [config]

    def choose_modules(self):
        """ TODO the docstring documentation
        """
        take = np.random.randint(0, self.pipe_length)

        preprocessors = self.preprocessors * (self.repetitions + 1)
        np.random.shuffle(preprocessors)
        return preprocessors[:take] + [np.random.choice(self.modelers)]

    def process_step(self, eval_result):
        """ TODO the docstring documentation
        """
        self.curr_eval = eval_result[0][1] or 0
        if self.curr_eval is not None \
                and self.curr_eval > self.best_eval:
            self.best_eval = self.curr_eval
            self.best_pipe = self.curr_pipe

    def get_best_pipeline(self):
        """ TODO the docstring documentation
        """
        return Pipeline(self.best_pipe)

    def get_current_eval(self):
        """ TODO the docstring documentation
        """
        return self.curr_eval

    def get_best_eval(self):
        """ TODO the docstring documentation
        """
        return self.best_eval

    def eval(self, pip_config, data):
        internal = cfg(Pipeline, configs=[
            cfg(CV, split='cv', steps=10, random_state=self.random_state),
            pip_config,
            cfg(Metric, function='accuracy')
        ], random_state=self.random_state)

        pip = Pipeline(config=cfg(
            Pipeline,
            configs=[
                cfg(
                    Iterator, configs=[internal], reduce=cfg(Reduce, field='r')
                ),
                cfg(Summ, field='s', function='mean')
            ],
            random_state=self.random_state),
            storage_settings=self.storage_settings_for_components
        )

        return pip, (pip.apply(data).s, pip.use(data).s)

    def _eval(self, component, data):
        # Start CV from beginning.
        self.cv = CV(self.cvargs)
        validation = self.cv

        result = {
            'measure_train': [],
            'measure_test': []
        }

        while True:
            train = validation.apply(data)
            test = validation.use(data)

            # TODO already did:
            #  ALERT!  apply() returns accuracy on the transformed set,
            #  not on the training set. E.g. noise reduction produces a smaller
            #  set to be evaluated by the model.
            #  We should use() the component on training data, if we want
            #  the training accuracy. So I discarded the result of apply()
            #  and added a new use().
            component.apply(train)
            output_train = component.use(train)
            output_test = component.use(test)

            if not (output_test and output_train):
                return None, None

            measure_train = output_train and self._metric(output_train)
            measure_test = output_test and self._metric(output_test)
            result['measure_train'].append(measure_train)
            result['measure_test'].append(measure_test)

            validation = validation.next()
            if validation is None:
                break

        return (
            self._summary(result['measure_train']),
            self._summary(result['measure_test'])
        )


def cfg(component, **kwargs):
    kwargs['class'] = component.__name__
    kwargs['module'] = component.__module__
    return kwargs
