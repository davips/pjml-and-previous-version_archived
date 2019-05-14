from paje.composer.composer import Composer
from paje.composer.pipeline import Pipeline
from paje.base.hps import HPTree

class Switch(Composer):

    def instantiate_impl(self):
        """
        The only parameter is dics with the dic of each component.
        :param dics
        :return:
        """
        if 'dics' in self.dic:
            dics = self.dic['dics']

        self.components = self.components.copy()

        component_idx = dics[0]["component"]
        component_idx = int(component_idx.split("_")[0])

        dic = dics[0].copy()
        dic['random_state'] = self.random_state
        del dic["component"]
        print(dic)

        self.components = [self.components[component_idx].instantiate(**dic)]

        # zipped = zip(range(0, len(self.components)), dics)
        # for idx, dic in zipped:
        #     if isinstance(self.components[idx], Composer):
        #         dic = {'dics': dic.copy()}
        #     dic['random_state'] = self.random_state
        #     self.components[idx] = self.components[idx].instantiate(**dic)

    def forest(self, data=None):  # previously known as hyperpar_spaces_forest
        forest = []
        idx = -1
        for component in self.components:
            idx += 1
            # if isinstance(component, Pipeline):
            #     aux = list(map(
            #         lambda x: x.forest(data),
            #         component.components
            #     ))
            #     tree = aux
            # else:
            tree = component.forest(data)
            comp_hptree = HPTree({"component": ['c', ["{0}_{1}".format(
                idx, component.__class__.__name__)]]}, [tree])
            forest.append(comp_hptree)

        return HPTree({}, children=forest)

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in
                self.components]
        return "Switch {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n'\
               + depth + "}"


