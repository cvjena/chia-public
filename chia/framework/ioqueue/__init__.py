import multiprocessing
import queue
import threading
import time
from typing import Any, Callable, Generator, Union


def make_generator_faster(
    gen: Callable[..., Generator[Any, None, None]],
    method: str,
    max_buffer_size: int = 100,
):
    if method == "threading":
        return make_generator_faster_threading(gen, max_buffer_size)
    elif method == "multiprocessing":
        return make_generator_faster_multiprocessing(gen, max_buffer_size)
    elif method == "synchronous":
        return gen()
    else:
        raise ValueError(f"Unknown method {method}")


def _producer_main(
    q_: Union[queue.Queue, multiprocessing.Queue],
    gen: Callable[..., Generator[Any, None, None]],
) -> None:
    gen_instance = gen()
    for item in gen_instance:
        q_.put(item)

    print("Producer done.")
    q_.put("THE_END")


def _consumer_generator(
    q_: Union[queue.Queue, multiprocessing.Queue]
) -> Generator[Any, None, None]:
    while True:
        try:
            item = q_.get(timeout=1.0)
            if item == "THE_END":
                print("Consumer done.")
                return
            else:
                yield item
        except queue.Empty:
            print("Empty queue!")
            time.sleep(1.0)


def make_generator_faster_threading(
    gen: Callable[..., Generator[Any, None, None]], max_buffer_size: int = 100
):
    q = queue.Queue(maxsize=max_buffer_size)
    producer_thread = threading.Thread(target=_producer_main, args=(q, gen))

    producer_thread.start()
    return _consumer_generator(q)


def make_generator_faster_multiprocessing(
    gen: Callable[..., Generator[Any, None, None]], max_buffer_size: int = 100
):
    q = multiprocessing.Queue(maxsize=max_buffer_size)
    producer_process = multiprocessing.Process(target=_producer_main, args=(q, gen))

    producer_process.start()
    return _consumer_generator(q)
