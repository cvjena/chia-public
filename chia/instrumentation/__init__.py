_current_context = None


class InstrumentationContext:
    def __init__(self, description, observers=None):
        self._description = description
        self._observers = observers if observers is not None else []
        self._local_step = None

    def __enter__(self):
        global _current_context
        self._parent_context = _current_context
        _current_context = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_context
        _current_context = self._parent_context

    def _update_local_step(self, local_step):
        self._local_step = local_step

    def report(self, metric, value, local_step=None):
        if local_step is not None:
            self._update_local_step(local_step)

        for observer in self._observers:
            observer.report(metric, value, self._local_step)

        if self._parent_context is not None:
            self._parent_context.report(f'{self._description}.{metric}', value)


def report(metric, value, local_step=None):
    if _current_context is not None:
        _current_context.report(metric, value, local_step)
    else:
        raise ValueError('Cannot report without Instrumentation Context')


class PrintObserver:
    def __init__(self, prefix=''):
        self._prefix = prefix

    def report(self, metric, value, local_step):
        description_string = f'{self._prefix:s}{metric:s}'
        if local_step is not None:
            print(f'{description_string:40s} @ {local_step:6d}: {value}')
        else:
            print(f'{description_string:49s}: {value}')
