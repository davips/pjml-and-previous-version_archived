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

    @property
    def result(self):
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
                item, acc = self.step_func(item, acc)
                yield item
        finally:
            self._result = self.summ_func(acc)
