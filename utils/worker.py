from __future__ import annotations

import sys
import logging
import inspect
from threading import Thread

from utils.queues import RequestQueue

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


def get_arguments(function, kwargs, substitute: dict = None):
    """Get the arguments of a function."""
    if substitute is not None:
        kwargs = kwargs.copy()
        for key, value in substitute.items():
            if key not in kwargs:
                continue
            kwargs[value] = kwargs[key]
            del kwargs[key]
    args = inspect.getfullargspec(function).args
    return {arg: kwargs[arg] for arg in args if arg in kwargs}


class DefaultDict(dict):
    def __missing__(self, key): return f'<{key}?>'


class EmptyIterator:
    def __iter__(self): return self

    def __next__(self): return {}


class KeepThemComing(Exception):
    pass


class Worker:
    def __init__(self, function, names=(), output=True, startup=None, cleanup=None, **kwargs):
        """
        This object is used to run a function in a separate thread.
        Before starting it, this object must be called with a queue of dicts to get the kwargs from.

        :param function: The function to be called repeatedly.
        :param names: The names of the results.
        :param output: If True, the results of the function are put in a queue.
        :param kwargs: Arguments to always be passed to the function.
        """
        self.function: callable = function
        if output:
            self.names: tuple[str] = tuple(names) or function.__dict__.get('names', [function.__name__])
        self.kwargs: dict = kwargs
        self.output: bool = output
        self.startup: callable = startup
        self.cleanup: callable = cleanup

        self.queue_in: RequestQueue | None = None
        self.queue_out: RequestQueue | None = None
        self.previous_elements = []
        self.substitute = {}

        WORKERS.add(self)

    def __call__(self, queue_in: RequestQueue | EmptyIterator, substitute: dict = None) -> RequestQueue | None:
        """
        :param queue_in: The queue to get the kwargs from.
        :return: The queue to put the results in.
        """
        self.queue_in = queue_in
        self.substitute = substitute or {}
        if self.queue_out is None and self.output:
            self.queue_out = RequestQueue()
            return self.queue_out

    def start(self, sparate_thread=True):
        """Starts calling the function in a separate thread by default."""
        if self.queue_in is None:
            raise ValueError("Queue not set.")

        if sparate_thread:
            Thread(target=self.run).start()
        else:
            self.run()

    def run(self):
        if self.startup is not None:
            self.startup()

        for element in self.queue_in:
            try:
                # This function does the actual work. It doesn't know about the queue.
                results = self.function(**get_arguments(self.function, self.kwargs | element, self.substitute))
            # Caching StopIteration allows an iterator's __next__ to be the function.
            except StopIteration:
                break
            except ValueError as e:
                # This format_map allows the functionn to print informative errors without knowing the iformation.
                log.debug(str(e).format_map(DefaultDict(self.kwargs | element)))
                continue
            # If the function needs multiple following values it will raise a KeepThemComming exception.
            except KeepThemComing as e:
                # If those values have to be returned, they are saved in the previous_elements list.
                if self.queue_out is not None and e.args and e.args[0]:
                    self.previous_elements.append(element)
                continue

            if self.queue_out is None: continue

            if self.previous_elements:
                if not isinstance(results, list):
                    raise ValueError(f"Function {self.function.__name__} must return a list of results after skipping"
                                     f"mutiple elements, not {type(results)}.")
                for result in results:
                    if isinstance(result, tuple) and len(result) != len(self.names) or len(self.names) != 1:
                        raise ValueError(f"Function {self.function.__name__} must return "
                                         f"{len(self.names)} elements, not {len(result)}.")
                    resultsDict = dict(zip(self.names, result if isinstance(result, tuple) else (result,)))
                    self.queue_out.append(self.previous_elements.pop(0) | resultsDict)
                self.previous_elements = []
            else:
                if isinstance(results, tuple) and len(results) != len(self.names) or len(self.names) != 1:
                    raise ValueError(f"Function {self.function.__name__} must return "
                                     f"{len(self.names)} elements, not {len(results)}.")
                resultsDict = dict(zip(self.names, results if isinstance(results, tuple) else (results,)))
                self.queue_out.append(element | resultsDict)

        if self.queue_out is not None:
            self.queue_out.off()

        if self.cleanup is not None:
            self.cleanup()


WORKERS = set()
