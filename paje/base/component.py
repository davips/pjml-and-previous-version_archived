from abc import ABC, abstractmethod


class Component(ABC):
    """Todo the docs string
    """

    def __init_subclass__(cls, **kwargs):
        cls.show_warnings = True

    @abstractmethod
    def apply(self, data):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def use(self, data):
        """Todo the doc string
        """
        pass

    # @abstractmethod
    def explain(self, X):
        """Explain prediction/transformation for the given instances.
        """
        raise NotImplementedError("Should it return probability distributions, rules?")

    @classmethod
    @abstractmethod
    def hps_impl(cls, data=None):
        """Todo the doc string
        """
        pass

    @classmethod
    def hps(cls, data=None):
        hps = cls.hps_impl(data)
        dic = hps.dic
        try:
            for k in dic:
                t = dic[k][0]
                v = dic[k][1]
                if t == 'c' or t == 'o':
                    if not isinstance(v, list):
                        print('Categorical and ordinal hyperparameters need a list of values: ' + str(k))
                        exit(0)
                else:
                    if len(dic[k]) != 3:
                        print('Real and integer hyperparameters need a limit with two values: ' + str(k))
                        exit(0)
        except:
            print('Problems with hyperparameter space: ' + str(dic))
            exit(0)
        return hps

    @classmethod
    def print_hps(cls, data=None, depth=''):
        tree = cls.hps(data)
        print(depth + str(tree.dic))
        depth += '     '
        for child in tree.children:
            cls.print_hps(child, depth)

    @classmethod
    def check_data(cls, data):
        if data is None:
            print(cls.__name__ + ' needs a dataset to be able to estimate maximum values for some hyperparameters.')
            exit(0)
