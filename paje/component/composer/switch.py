from paje.component.composer.composer import Composer
from paje.base.hps import HPTree


class Switch(Composer):
    def build_impl(self):
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
        dic['random_state'] = self.dic['random_state']  # TODO: check this
        del dic["component"]
        # TODO: is switch ready?

        self.components = [self.components[component_idx].build(**dic)]

    def tree(self, data=None):
        forest = []
        idx = -1
        for component in self.components:
            idx += 1
            tree = component.tree(data)
            comp_hptree = HPTree({"component": ['c', ["{0}_{1}".format(
                idx, component.name)]]}, [tree])
            forest.append(comp_hptree)

        return HPTree({}, children=forest, name=self.name)

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in
                self.components]
        return self.name + " {\n" + newdepth + \
               ("\n" + newdepth).join(str(x) for x in strs) + '\n' + depth + "}"
