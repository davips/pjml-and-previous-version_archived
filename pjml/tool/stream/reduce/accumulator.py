from dataclasses import dataclass
from typing import Callable, List, Iterator, Tuple

from pjdata.content.data import Data
from pjdata.types import Field


@dataclass
class Result:
    value: float


@dataclass
class Accumulator:
    """Cumulative iterator that returns a final/result value.

    The enclosed iterator should be finite."""

    iterator: Iterator[Data]
    start: List[Field]
    step_func: Callable[[Data, List[Field]], Tuple[Data, List[Field]]]
    summ_func: Callable[[List[Field]], Field]
    stream_exception = False

    @property
    def result(self):
        from pjml.tool.stream.reduce.summ import InterruptedStreamException
        if self.stream_exception:
            raise InterruptedStreamException
        try:
            return self._result
        except AttributeError as e:
            print("Stream not consumed!\nHINT: The result of summarizers are only accessible after Reduce.")
            # exit()
            raise e from None

    def __iter__(self):
        acc = self.start.copy()
        try:
            for item in self.iterator:
                if not self.stream_exception:
                    item, acc = self.step_func(item, acc)
                    if acc is None:
                        self.stream_exception = True
                yield item
        finally:
            if not self.stream_exception:
                self._result = self.summ_func(acc)
