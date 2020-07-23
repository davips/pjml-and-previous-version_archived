from functools import partial

from time import sleep

from cururu.worker import Worker, Nothing
from pjml.tool.abs.mixin.timing import withTiming


def f(a, b):
    print('start', a, b)
    sleep(a + b)
    print('end', a, b)
    print()
    return Nothing


start = withTiming._clock()

w = Worker()
w.put((f, {'a': 2, 'b': 1}))
print(1)
w.put((f, {'a': 0, 'b': 1}))
print(2)
w.put((f, {'a': 2, 'b': 0}))
print(3)

print('Tempo: ', '{:.2f}'.format(withTiming._clock() - start))
w.join()
print('Tempo tot: ', withTiming._clock() - start)
