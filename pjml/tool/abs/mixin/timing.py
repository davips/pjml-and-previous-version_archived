import os
import signal
from contextlib import contextmanager
import time


class withTiming:
    """Management of time.
    """

    @staticmethod
    def _cpu():
        """CPU time.

        Returns
        -------
            Sum of all SO times except wall time.
        """
        #  return time.process_time()  # Does not include children processes.
        t = os.times()
        return t[0] + t[1] + t[2] + t[3]

    @staticmethod
    def _clock():
        """Wall clock time.

        Returns
        -------
            Ellapsed time.
        """
        return time.monotonic()

    def _limit_by_time(self, function, data, max_time, **kwargs):
        if max_time is None:
            return function(data, **kwargs)
        else:
            with self._time_limit(max_time):
                return function(data, **kwargs)

    @staticmethod
    @contextmanager
    def _time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


class TimeoutException(Exception):
    pass
