from paje.composer.composer import Composer
from paje.base.hps import HPTree
import copy


class Pipeline(Composer):
    def fields_to_store_after_use(self):
        return self.components[len(self.components) - 1] \
            .fields_to_store_after_use()

    def fields_to_keep_after_use(self):
        return self.components[len(self.components) - 1] \
            .fields_to_keep_after_use()

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
            self.components[idx] = self.components[idx].build(**dic)
            # component.instantiate(**dic)

    def set_leaf(self, tree, f):
        # TODO: verify why there is a excess of EndPipelines
        # print('setleaf',tree.name)
        if len(tree.children) > 0:
            for child in tree.children:
                self.set_leaf(child, f)
        else:
            if not (tree.name.startswith('End')
                    and tree.tmp_uuid == self.tmp_uuid):
                tree.children.append(f())

    def tree_impl(self, data=None):
        if self.mytree is None:
            trees = []
            for i in range(0, len(self.components)):
                # TODO: Why were we using deepcopy here?
                tree = copy.copy(self.components[i]).tree(data)
                trees.append(tree)

            self.set_leaf(trees[len(trees) - 1], lambda:
            HPTree({}, [], name='End' + self.name, tmp_uuid=self.tmp_uuid))
            for i in reversed(range(1, len(trees))):
                self.set_leaf(trees[i - 1], lambda: trees[i])

            self.mytree = HPTree({}, [trees[0]], self.name)

        return self.mytree

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in self.components]
        return self.name + " {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n' \
               + depth + "}"
