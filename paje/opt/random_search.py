# class Optimization(object):
#     """ A wrapper for all optmization method. """
#     def __init__(self, space, method="RS", ...):
#         self.space = space
#
#     def fmin():
#         pass
#
#
#
import numpy as np

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
        nro_branches = len(space.children)
        conf.update(space.get_data())

        if nro_branches:
            aux = np.random.randint(nro_branches)
            self.__get_random_attr(space.children[aux], conf)


    def fmin(self, objective):
        best_conf = self.get_random_attr()
        best_value = objective(**best_conf)
        for t in range(1, self.max_iter):
            conf = self.get_random_attr()
            value = objective(**conf)

            print(best_value)
            if value < best_value:
                best_value = value
                best_conf = conf

        return best_value, best_conf


class Node(object):

    def __init__(self, name=None, parent=None):
        self.children = []
        self.data = {}

        self.name = ""
        if name != None:
            self.name = name

        if(parent != None):
            parent.add_child(self)

    def add_child(self, child):
        self.children.append(child)

    def add_content(self, name, content):
        self.data[name] = content

    def get_value(self,k):
        c_type, s, e, f = self.data[k]
        value = None

        if c_type == 'c':
            value = f[np.random.randint(0,len(f))]
        elif c_type == 'z':
            value = int(np.rint(((e-s)*f()) - s))
        elif c_type == 'r':
            value = ((e-s)*f()) - s
        elif c_type == 'f':
            value = f()

        return value

    def get_data(self):
        return {k:self.get_value(k) for k in self.data.keys()}

    def get_content():
        return {k:get_value(**self.data[k]) for k in self.data.keys()}

    def print(self, space="  ", data=False):
        print("{0}|__ {1}".format(space, self.name))

        if data == True:
            for k in self.data.keys():
                print("{0}   {1}".format(space, k))

        for c in self.children:
            c.print(space=space+"  ", data=data)


class HPSpace(Node):

    def __init__(self, name=None):
        Node.__init__(self, name=name, parent=None)

    def add_axis(cls, branch, axis_name, axis_type, axis_min, axis_max, axis_value):
        branch.add_content(axis_name, [axis_type, axis_min, axis_max, axis_value])

    def add_branch(cls, father, name=None):
        return Node(name=name, parent=father)

def my_func():
    return np.random.randint(1, 10)

def making_space():
    hp = HPSpace(name="root")
    hp.add_axis(hp, "x1", 'c', 0, 5, ['5', '10', '15', '20', '25'])

    b1 = hp.add_branch(hp, "b1")
    hp.add_axis(b1, "x2", 'r', 0, 10, np.random.ranf)
    hp.add_axis(b1, "x3", 'z', -2, 10, np.random.ranf)

    b2 = hp.add_branch(hp, "b2")
    hp.add_axis(b2, "x4", 'f', None, None, my_func)

    hp.print(data=True)

    return hp


def objective(*argv, **kwargs):
    print("*argv --> {0} \n **kwargs --> {1}".format(argv, kwargs))

    aux = 10000
    x3 = kwargs.get('x3')
    x1 = kwargs.get('x1')
    x4 = kwargs.get('x4')

    if x3 != None:
        x2 = kwargs.get('x2')
        aux = int(x1) + x2 + x3 + 1
    elif x4 != None:
        aux = int(x1) * x4
    else:
        print("Some error occur")


    return aux


sr = RandomSearch(making_space())
result = sr.fmin(objective)
print("Best fmin = {0}\nConf = {1}\n".format(result[0], result[1]))
