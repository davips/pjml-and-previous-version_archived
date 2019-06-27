# from functools import reduce
#
# from paje.base.data import Data
# from paje.composer.pipeline import Pipeline
# import numpy as np
#
#
# class Concat(Pipeline):
#     def __init__(self, components=None, matrices=None, direction='horizontal',
#                  storage=None, show_warns=True):
#         super().__init__(storage=storage, show_warns=show_warns)
#         if matrices is None:
#             matrices = ['X']
#         if components is None:
#             components = []
#         self.components = components
#         self.matrices = matrices
#         self.axis = 1 if direction == 'horizontal' else 0
#         self.random_state = 0
#         self.myforest = None
#         self.model = 42
#
#     def apply_impl(self, data):
#         datas = []
#         for component in self.components:
#             datas.append(component.apply(data))
#             if component.failed:
#                 raise Exception('Applying subcomponent failed! ', component)
#         return self.concat_list(datas)
#
#     def use_impl(self, data):
#         datas = []
#         for component in self.components:
#             datas.append(component.use(data))
#             if component.failed:
#                 raise Exception('Using subcomponent failed! ', component)
#         return self.concat_list(datas)
#
#     # TODO: check if all data shapes match
#     def concat(self, a, b, attr):
#         return np.concatenate((a.__getattribute__(attr),
#                                b.__getattribute__(attr)), axis=self.axis)
#
#     def concat_list(self, datas):
#         dic = {}
#         for attr in self.matrices:
#             M = reduce(lambda A, B: self.concat(A, B, attr), datas)
#             dic[attr] = M
#         return Data(name='concat', **dic) #TODO: choose a proper name
