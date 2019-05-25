from paje.composer.composer import Composer
from paje.base.hps import HPTree
import copy


class Pipeline(Composer):

    # @profile
    def build_impl(self):
        """
        The only parameter is dics with the dic of each component.
        :param dics
        :return:
        """
        dics = [{} for _ in self.components]  # Default value

        if 'dics' in self.dic:
            dics = self.dic['dics']
        # if 'random_state' in self.dic:
        #     self.random_state = self.dic['random_state']
        self.components = self.components.copy()
        # exit(0)
        zipped = zip(range(0, len(self.components)), dics)
        for idx, dic in zipped:
            # TODO: setar showwarns?
            # if isinstance(self.components[idx], Composer):
            #     dic = {'dics': dic.copy()}

            dic = dic.copy()
            dic['random_state'] = self.random_state
            # print('comp',self.components[idx])
            # print('dic', dic)
            # self.components[idx].memoize = self.memoize
            self.components[idx] = self.components[idx].build(**dic)
            # component.instantiate(**dic)

    # @profile
    def set_leaf(self, tree, f):
        # TODO: verify why there is a excess of EndPipelines
        # print('setleaf',tree.name)
        if len(tree.children) > 0:
            for child in tree.children:
                self.set_leaf(child, f)
        else:
            # if tree.name is not 'EndPipeline':
            tree.children.append(f())

    # @profile
    def tree_impl(self, data=None):
        # forest = []
        if self.myforest is None:
            # for component in self.components:
            # self.myforest = self.components[0].forest(data)
            # tree = self.myforest
            # trees = [copy.deepcopy(i.forest(data)) for i in self.components]
            trees = []
            for i in range(0, len(self.components)):
                # TODO: Why were we using deepcopy here?
                tree = copy.copy(self.components[i]).tree(data)
                # tree.name = "{0}_{1}".format(
                #     i, self.components[i].name)
                trees.append(tree)

            for i in reversed(range(1, len(trees))):
                self.set_leaf(trees[i - 1], lambda: trees[i])

                # if isinstance(component, Pipeline):
                #     aux = list(map(
                #         lambda x: x.forest(data),
                #         component.components
                #     ))
                #     tree = aux
                # else:
                # tree = component.forest(data)
                # forest.append(tree)
            self.myforest = HPTree({}, [trees[0]], self.name)
            self.set_leaf(trees[len(trees) - 1],
                          lambda: HPTree({}, [], 'End'+self.name))
        return self.myforest

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in self.components]
        return self.name + " {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n' \
               + depth + "}"
