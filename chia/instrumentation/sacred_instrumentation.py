""" sacred_instrumentation: For this file you need to install the sacred pip package."""
import sacred
import threading
import os

from chia import instrumentation
from chia import configuration


class SacredObserver(instrumentation.InstrumentationObserver):
    def __init__(self, prefix=''):
        instrumentation.InstrumentationObserver.__init__(self, prefix)

        # Configuration
        with configuration.ConfigurationContext(self.__class__.__name__):
            self._mongo_observer = sacred.observers.MongoObserver.create(url=configuration.get('mongo_url', next(
                open(os.path.expanduser("~/work/experiments/sacred/mongourl")))),
                                                                         db_name=configuration.get('mongo_db_name',
                                                                                                   'sacred'))
        self.sacred_experiment = None
        self.sacred_run = None

        self.done = None
        self.in_run_func = None
        self.sacred_thread = None

    def report(self, metric, value, steps, contexts):
        assert self.sacred_run is not None
        description_string, steps_string = self.build_description_string_from(metric, contexts, steps)
        self.sacred_run.log_scalar(description_string, value, steps_string)

    def on_context_enter(self):
        super().on_context_enter()

        self.sacred_experiment = sacred.Experiment(self._prefix)
        self.sacred_experiment.observers.append(self._mongo_observer)

        def experiment_main(_run):
            self.sacred_run = _run
            self.in_run_func.set()
            self.done.wait()

        self.sacred_experiment.main(experiment_main)

        self.in_run_func = threading.Event()
        self.done = threading.Event()
        self.sacred_thread = threading.Thread(target=self.sacred_experiment.run)
        self.sacred_thread.start()
        self.in_run_func.wait()

    def on_context_exit(self):
        super().on_context_exit()
        self.done.set()
        self.sacred_thread.join()


