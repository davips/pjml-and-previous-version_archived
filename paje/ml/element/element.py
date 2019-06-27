from paje.ml.component import Component


class Element(Component):
    def modifies(self):
        print('ALERT: implement Element.modifies() correctly to avoid '
              'useless traffic!')
        print(' Evil plan: inspect.getsource( ... ) ')
        # inspect.getsource(
        return ['x', 'y', 'z']
