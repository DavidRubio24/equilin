import threading


class LockObject:
    def __init__(self, obj): self.object = obj

    def __enter__(self):
        default_lock = threading.Lock()
        with LockObject.Lock:
            lock, count = LockObject.LOCKS.get(self.object, (default_lock, set()))
            count.add(self)
            LockObject.LOCKS[self.object] = (lock, count)
        lock.acquire()

    def __exit__(self, *_):
        with LockObject.Lock:
            lock, count = LockObject.LOCKS[self.object]
            count.discard(self)
            if not count:
                del LockObject.LOCKS[self.object]
        lock.release()


LockObject.LOCKS = {}
LockObject.Lock = threading.Lock()
