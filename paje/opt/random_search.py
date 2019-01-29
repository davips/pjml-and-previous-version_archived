import numpy as np
from paje.opt.hp_space import HPSpace

class RandomSearch():
    """ Random Search method """

    def __init__(self, space, max_iter=10):
        """ Come thing

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.



        """
        self.space = space
        self.max_iter = max_iter


    def get_random_attr(self):
        conf = {}
        self.__get_random_attr(self.space, conf)

        return conf


    def __get_random_attr(self, space, conf):
        nro_branches = space.nro_branches()
        conf.update(space.get_data())

        if nro_branches:
            aux = np.random.randint(nro_branches)
            self.__get_random_attr(space.get_branch(aux), conf)


    def fmin(self, objective):
        best_conf = self.get_random_attr()
        best_value = objective(**best_conf)
        print(best_value)
        for t in range(1, self.max_iter):
            conf = self.get_random_attr()
            value = objective(**conf)

            print(value)
            if value < best_value:
                best_value = value
                best_conf = conf

        return best_value, best_conf
