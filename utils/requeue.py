import time

from lockobject import LockObject


class ReQueue:
    NEXT = 1
    LAST = 2
    ALL = 3

    def __init__(self):
        self.items = []

        self.more = True
        self.next_item = 0

        self.observers: set[ReQueueIterator] = set()

    def __iter__(self): return ReQueueIterator(self)

    def __call__(self, mode=NEXT): return ReQueueIterator(self, mode)

    def append(self, item):
        self.items.append(item)

    def off(self): self.more = False


class ReQueueIterator:
    def __init__(self, requeue, mode=ReQueue.NEXT):
        with LockObject(requeue):
            self.next_item: int = requeue.next_item
            requeue.observers.add(self)
        self.requeue: ReQueue = requeue
        self.mode = mode

    def __lt__(self, other): return self.next_item < other.next_item

    def __iter__(self): return self

    def __next__(self):
        # Make sure we are not trynna access an item that is not there. Shouldn't happen.
        next_item = max(self.next_item, self.requeue.next_item)
        with LockObject(self.requeue):
            self.next_item = next_item

        # Wait until the next item is available.
        while len(self.requeue.items) <= self.next_item - self.requeue.next_item:
            # No more items already read.
            if not self.requeue.more:
                # No more items to read: end of iterator.
                with LockObject(self.requeue):
                    self.requeue.observers.discard(self)
                raise StopIteration()
            else:
                # Wait for next item.
                time.sleep(0)

        # Get the next item(s).
        # Depending on the mode, we return either the next item, the last item or all items.
        item = None
        next_item = self.next_item
        if self.mode == ReQueue.NEXT:
            item = self.requeue.items[self.next_item - self.requeue.next_item]
            next_item = self.next_item + 1
        elif self.mode == ReQueue.LAST:
            item = self.requeue.items[-1]
            next_item = len(self.requeue.items) + self.requeue.next_item
        elif self.mode == ReQueue.ALL:
            item = self.requeue.items[self.next_item - self.requeue.next_item:]
            next_item = len(self.requeue.items) + self.requeue.next_item

        # Update next_image. Deleting the image from the list if necessary.
        with LockObject(self.requeue):
            self.next_item = next_item
            min_observer = min(self.requeue.observers)
            del self.requeue.items[:min_observer.next_item - self.requeue.next_item]
            self.requeue.next_item = min_observer.next_item
        return item

    def __call__(self, mode=ReQueue.NEXT):
        self.mode = mode
        return self

    def __del__(self):
        with LockObject(self.requeue):
            self.requeue.observers.discard(self)
