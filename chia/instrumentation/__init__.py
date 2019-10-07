import time

_current_context = None


class InstrumentationTimer:
    def __init__(self, description):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        total_time = float(self.end_time - self.start_time)
        report(f'time_{self.description}', total_time)


class InstrumentationContext:
    def __init__(self, description, observers=None, take_time=False):
        self._description = description
        self._observers = observers if observers is not None else []
        self._local_step = None

        self.take_time = take_time

    def __enter__(self):
        global _current_context
        self._parent_context = _current_context
        _current_context = self

        if self.take_time:
            self.timer = InstrumentationTimer('total_time').__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.take_time:
            self.timer.__exit__(None, None, None)
        global _current_context
        _current_context = self._parent_context

    def update_local_step(self, local_step):
        self._local_step = local_step

    def report(self, metric, value, local_step=None):
        if local_step is not None:
            self.update_local_step(local_step)

        for observer in self._observers:
            observer.report(metric, value, self._local_step)

        if self._parent_context is not None:
            self._parent_context.report(f'{self._description}.{metric}', value)


def report(metric, value, local_step=None):
    if _current_context is not None:
        _current_context.report(metric, value, local_step)
    else:
        raise ValueError('Cannot report without Instrumentation Context')


def report_dict(mv_dict, local_step=None):
    if _current_context is not None:
        for metric, value in mv_dict.items():
            _current_context.report(metric, value, local_step)
    else:
        raise ValueError('Cannot report without Instrumentation Context')


def update_local_step(local_step):
    if _current_context is not None:
        _current_context.update_local_step(local_step)
    else:
        raise ValueError('Cannot update local step without Instrumentation Context')


class PrintObserver:
    def __init__(self, prefix=''):
        self._prefix = prefix

    def report(self, metric, value, local_step):
        description_string = f'{self._prefix:s}{metric:s}'
        if local_step is not None:
            print(f'{description_string:40s} @ {local_step:6d}: {value}')
        else:
            print(f'{description_string:49s}: {value}')



