from abc import ABC, abstractmethod

import numpy as np

from paje.component.element.preprocessing.supervised.instance.sampler.cv \
    import CV
from paje.evaluator.metrics import Metrics


class Evaluator(ABC):
    @abstractmethod
    def eval(self, component, data):
        pass


class EvaluatorClassif(Evaluator):
    _metrics = {
        'error': Metrics.error,
        'accuracy': Metrics.accuracy
    }

    _summaries = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std
    }

    def __init__(self,
                 metric='accuracy',
                 split='cv',
                 steps=10,
                 test_size=1 / 3,
                 summary='mean',
                 random_state=0):

        self.split = split
        self.metric = metric
        self.steps = steps
        self.test_size = test_size
        self.summary = summary
        self.random_state = random_state

        if self.split == 'cv':
            self.cvargs = {
                'split': self.split,
                'steps': self.steps,
                'random_state': self.random_state
            }
        elif self.split == 'loo':
            self.cvargs = {}
        elif self.split == 'holdout':
            self.cvargs = {
                'split': self.split,
                'steps': self.steps,
                'test_size': self.test_size,
                'random_state': self.random_state
            }
        else:
            raise ValueError(
                "split must be 'cv' or 'loo' or 'holdout'"
            )

        if self.metric in self._metrics:
            self._metric = self._metrics[self.metric]
        else:
            raise ValueError(
                "metric must be 'accuracy' or 'error'"
            )

        if self.summary in self._summaries:
            self._summary = self._summaries[self.summary]
        else:
            raise ValueError(
                "summary must be 'mean' or 'median' or 'std'"
            )

    def describe(self):
        return {
            'module': self.__class__.__module__,
            'name': self.__class__.__name__,
            'args_set': {
                'metric': self.metric,
                'split': self.split,
                'steps': self.steps,
                'test_size': self.test_size,
                'summary': self.summary,
                'random_state': self.random_state
            }
        }

    def eval(self, component, data):
        # Start CV from beginning.
        self.cv = CV().build(**self.cvargs)
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
