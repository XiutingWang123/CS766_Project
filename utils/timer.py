import time
import datetime


class Timer(object):
    """
    Build a simple timer to trace training time and data loading time
    """

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def start_timer(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def end_timer(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iter, max_iter):
        if iter == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) / iter * (max_iter - iter)
        return str(datetime.timedelta(seconds=int(self.remain_time)))