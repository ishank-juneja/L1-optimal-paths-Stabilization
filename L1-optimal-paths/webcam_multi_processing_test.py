from multiprocessing import Process, Pipe
import numpy as np

Size = 128


def producer(conn):
    print("Child entered")
    window1 = np.arange(Size * Size).reshape((Size, Size))
    print("Array created")
    conn.send(window1)
    print("Data added to Pipe")
    return


if __name__ == '__main__':
    conn1, conn2 = Pipe()
    window_reader = Process(target=producer, args=(conn2,))
    window_reader.start()
    window_reader.join()
    print("Child process exited, entered parent")
    # Acting like a consumer
    a = conn1.recv()
    print(a)
