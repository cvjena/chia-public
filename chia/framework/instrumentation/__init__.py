import abc
import json
import random
import time

from chia.framework import configuration

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
        report(f"time_{self.description}", total_time)


class InstrumentationContext:
    def __init__(self, description, observers=None, take_time=False):
        self._description = description
        self._observers = observers if observers is not None else []
        self._local_step = None
        self.run_id = f"{random.getrandbits(32):08x}"

        self.take_time = take_time
        self.stored_result = None

    def __enter__(self):
        global _current_context
        self._parent_context = _current_context
        _current_context = self

        if self.take_time:
            self.timer = InstrumentationTimer("total_time").__enter__()

        for observer in self._observers:
            observer.on_context_enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for observer in self._observers:
            observer.on_context_exit(self.stored_result, self.run_id, exc_val)

        if self.take_time:
            self.timer.__exit__(None, None, None)
        global _current_context
        _current_context = self._parent_context

    def update_local_step(self, local_step):
        self._local_step = local_step

    def report(
        self, metric, value, local_step=None, inner_steps=None, inner_contexts=None
    ):
        # Process step info
        if local_step is not None:
            self.update_local_step(local_step)

        if inner_steps is None:
            inner_steps = []

        steps = [self._local_step] + inner_steps

        # Process context info
        if inner_contexts is None:
            inner_contexts = []

        contexts = [self._description] + inner_contexts

        for observer in self._observers:
            observer.report(metric, value, steps=steps, contexts=contexts)

        if self._parent_context is not None:
            self._parent_context.report(
                metric, value, inner_steps=steps, inner_contexts=contexts
            )

    def store_result(self, result):
        self.stored_result = result

    def update_run_id(self, run_id):
        print(f"New run id: {run_id}")
        self.run_id = run_id


def report(metric, value, local_step=None):
    if _current_context is not None:
        _current_context.report(metric, value, local_step)
    else:
        raise ValueError("Cannot report without Instrumentation Context")


def report_dict(mv_dict, local_step=None):
    if _current_context is not None:
        for metric, value in mv_dict.items():
            _current_context.report(metric, value, local_step)
    else:
        raise ValueError("Cannot report without Instrumentation Context")


def store_result(result):
    if _current_context is not None:
        _current_context.store_result(result)
    else:
        raise ValueError("Cannot store result without Instrumentation Context")


def update_local_step(local_step):
    if _current_context is not None:
        _current_context.update_local_step(local_step)
    else:
        raise ValueError("Cannot update local step without Instrumentation Context")


def update_run_id(run_id):
    if _current_context is not None:
        _current_context.update_run_id(run_id)
    else:
        raise ValueError("Cannot update run ID without Instrumentation Context")


class InstrumentationObserver(abc.ABC):
    def __init__(self, prefix=""):
        self._prefix = prefix

    def on_context_enter(self):
        pass

    def on_context_exit(self, stored_result, run_id, stored_exception):
        pass

    @abc.abstractmethod
    def report(self, metric, value, steps, contexts):
        pass

    def build_description_string_from(self, metric, contexts, steps):
        step_strings = map(lambda step: str(step) if step is not None else "-", steps)
        steps_string = ".".join(step_strings)
        context_strings = map(
            lambda context: str(context) if context is not None else "-", contexts
        )
        contexts_string = ".".join(context_strings)
        description_string = f"{self._prefix:s}.{contexts_string:s}/{metric:s}"
        return description_string, steps_string


class PrintObserver(InstrumentationObserver):
    def report(self, metric, value, steps, contexts):
        description_string, steps_string = self.build_description_string_from(
            metric, contexts, steps
        )
        print(f"{description_string:49s} @ {steps_string:10s}: {value}")

    def on_context_exit(self, stored_result, run_id, stored_exception):
        if stored_result is not None:
            print(f"{self._prefix}#{run_id}: RESULT = {stored_result}")
        if stored_exception is not None:
            print(f"{self._prefix}#{run_id}: EXC = {stored_exception}")


class JSONResultObserver(InstrumentationObserver):
    def __init__(self, prefix=""):
        super().__init__(prefix)
        self._intermediate_results = []

    def report(self, metric, value, steps, contexts):
        self._intermediate_results += [
            {"metric": metric, "value": value, "contexts": contexts, "steps": steps}
        ]

    def on_context_exit(self, stored_result, run_id, stored_exception):
        config_dump = {
            "custom": configuration.dump_custom_dict(),
            "default": configuration.dump_default_dict(),
        }

        result_dict = {
            "prefix": self._prefix,
            "run_id": run_id,
            "stored_result": stored_result,
            "configuration": config_dump,
            "intermediate_results": self._intermediate_results,
            "stored_exception": str(stored_exception),
        }

        try:
            output = json.dumps(result_dict, indent=2)
        except Exception:
            output = str(result_dict)

        try:
            fname = f"storedresult-{self._prefix}-{run_id}.json"
            with open(fname, "w") as target:
                target.write(output)
        except Exception:
            print("WARNING: Could not store result: {str(ex}")
