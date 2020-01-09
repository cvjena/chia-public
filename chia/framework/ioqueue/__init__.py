import collections
import time
import threading
import multiprocessing
import queue


def make_generator_faster(gen, method, max_buffer_size=100):
    if method == "threading":
        return make_generator_faster_threading(gen, max_buffer_size)
    elif method == "multiprocessing":
        return make_generator_faster_multiprocessing(gen, max_buffer_size)
    elif method == "synchronous":
        return gen()


def make_generator_faster_threading(gen, max_buffer_size=100):
    def producer_thread_main(q_: queue.Queue):
        gen_instance = gen()
        for item in gen_instance:
            q_.put(item)

        print("Producer done.")
        q_.put("THE_END")

    def consumer_generator(q_: queue.Queue):
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

    q = queue.Queue(maxsize=max_buffer_size)
    producer_thread = threading.Thread(target=producer_thread_main, args=(q,))

    producer_thread.start()
    return consumer_generator(q)


def make_generator_faster_multiprocessing(gen, max_buffer_size=100):
    def producer_process_main(q_: multiprocessing.Queue):
        gen_instance = gen()
        for item in gen_instance:
            q_.put(item)

        print("Producer done.")
        q_.put("THE_END")

    def consumer_generator(q_: multiprocessing.Queue):
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

    q = multiprocessing.Queue(maxsize=max_buffer_size)
    producer_process = multiprocessing.Process(target=producer_process_main, args=(q,))

    producer_process.start()
    return consumer_generator(q)
