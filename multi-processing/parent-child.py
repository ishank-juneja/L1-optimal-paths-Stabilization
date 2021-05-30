from multiprocessing import Process, Queue
import numpy as np

Size = 127


def producer(mp_queue):
    print("Child entered")
    window1 = np.arange(Size * Size).reshape((Size, Size))
    print(window1.dtype)
    print("Array created")
    mp_queue.put(window1)
    print("Data added to Pipe")
    return


if __name__ == '__main__':
    q = Queue()
    window_reader = Process(target=producer, args=(q,))
    window_reader.start()
    window_reader.join()
    print("Child process exited, entered parent")
    # Acting like a consumer
    a = q.get()
    print(a)
