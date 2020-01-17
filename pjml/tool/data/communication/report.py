from pjml.tool.base.transformer import Transformer
from pjml.tool.common.invisible import Invisible
from pjml.util import flatten

import re

class Report(Invisible):
    """Report printer.

    $r prints 'r'
    {dataset.name} prints the dataset name
    {dataset.failure} prints the failure
    """

    def __init__(self, text='Default report r=$r'):
        Transformer.__init__(self, {'text': text}, text, deterministic=True)
        self.model = text
        self.text = text

    def _apply_impl(self, data):
        print('[apply] ', self._interpolate(self.text, data))
        return data

    def _use_impl(self, data):
        print('[use] ', self._interpolate(self.text, data))
        return data

    @classmethod
    def _interpolate(cls, text, data):
        def f(obj_match):
            return str(data.fields_safe(cls, obj_match.group(1)))

        p = re.compile(r'\$([a-zA-Z]+)')
        return cls._eval(p.sub(f, text), data)

    @classmethod
    def _eval(cls, text, data):
        txt = ''
        run = False
        expanded = [w.split('}') for w in ('_' + text + '_').split('{')]
        for seg in flatten(expanded):
            if run:
                txt += str(eval('data.' + seg))
            else:
                txt += seg
            run = not run
        return txt[1:][:-1]
