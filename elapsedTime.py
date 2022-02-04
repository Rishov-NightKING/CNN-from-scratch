import time
import threading


class ElapsedTimeThread(threading.Thread):
    """"Stoppable thread that prints the time elapsed"""

    def __init__(self):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        thread_start = time.time()
        while not self.stopped():
            print("\rElapsed Time: {:.3f} seconds".format(time.time() - thread_start), end="")
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(0.01)
