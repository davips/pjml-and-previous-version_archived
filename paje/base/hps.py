import random
import traceback


# class SNode(Node):
#     def __init__(self, name, children):
#         super().__init__(children=children)
#         self.name = name
#         self.hps = []
#
#
# # class ENode(Node):
# #     def __init__(self):
# #         super().__init__(children=[])
# #         self.hps = []
# #
# #
# # class INode(Node):
# #     def __init__(self, hps, children=None):
# #         children = [] if children is None else children
# #         super().__init__(children=children)
# #         self.hps = hps
# #
# #     def updated(self, children):
# #         return INode(self.hps, children=children)
# #
# #     def __str__(self, depth=''):
# #         rows = [str(self.hps)]
# #         for child in self.children:
# #             rows.append(child.__str__(depth + '   '))
# #         return depth + self.__class__.__name__ + '\n'.join(rows)
# #
# #     __repr__ = __str__


class Node:
    def __init__(self, hps, name=None, nested=None, children=None):
        self.children = [] if children is None else children
        self.name = name
        self.hps = hps
        self.nested = nested

    def updated(self, **kwargs):
        dic = {
            'hps': self.hps,
            'name': self.name,
            'nested': self.nested,
            'children': self.children
        }
        dic.update(kwargs)
        return Node(**dic)

    def sample(self):
        """TODO:
        """
        config = self._elem_hps_to_config(self)

        return config

    def _elem_hps_to_config(self, node):
        args = {}

        for hp in node.hps:
            try:
                args[hp.name] = hp.sample()
            except Exception as e:
                traceback.print_exc()
                print(e)
                print('Problems sampling: ', hp.name, hp)
                exit(0)

        if node.children:
            child = random.choice(node.children)

            config = self._elem_hps_to_config(child)
            args.update(config)

        return args

    def __str__(self, depth=''):
        rows = [str(self.hps)]
        for child in self.children:
            rows.append(child.__str__(depth + '   '))
        return depth + self.__class__.__name__ + '\n'.join(rows) \
               + str(self.nested)

    __repr__ = __str__


class ConfigSpace(Node):
    pass
