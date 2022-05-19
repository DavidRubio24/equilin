import time

from ppg08.lockobject import LockObject


class ReQueue:
    NEXT = 1
    LAST = 2
    ALL = 3

    def __init__(self):
        self.items = []

        self.more = True
        self.removed = 0

        self.observers: set[ReQueueIterator] = set()

    def __iter__(self): return ReQueueIterator(self)

    def __call__(self, mode=NEXT): return ReQueueIterator(self, mode)

    def append(self, item):
        self.items.append(item)

    def off(self): self.more = False


class ReQueueIterator:
    def __init__(self, requeue, mode=ReQueue.NEXT):
        with LockObject(requeue):
            self.next_item: int = requeue.removed
            requeue.observers.add(self)
        self.requeue: ReQueue = requeue
        self.mode = mode

    def __lt__(self, other): return self.next_item < other.next_item

    def __iter__(self): return self

    def __next__(self):
        # Make sure we are not trynna access an item that is not there. Shouldn't happen.
        next_item = max(self.next_item, self.requeue.removed)
        with LockObject(self.requeue):
            self.next_item = next_item

        # Wait until the next item is available.
        while len(self.requeue.items) <= self.next_item - self.requeue.removed:
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
        current_item = next_item
        if self.mode == ReQueue.NEXT:
            current_item = self.next_item
            item = self.requeue.items[self.next_item - self.requeue.removed]
            next_item = self.next_item + 1
        elif self.mode == ReQueue.LAST:
            current_item = len(self.requeue.items) + self.requeue.removed - 1
            item = self.requeue.items[-1]
            next_item = len(self.requeue.items) + self.requeue.removed
        elif self.mode == ReQueue.ALL:
            current_item = len(self.requeue.items) + self.requeue.removed - 1
            item = self.requeue.items[self.next_item - self.requeue.removed:]
            next_item = len(self.requeue.items) + self.requeue.removed

        # Update next_image. Deleting the image from the list if necessary.
        with LockObject(self.requeue):
            self.next_item = next_item
            min_observer = min(self.requeue.observers)
            del self.requeue.items[:min_observer.next_item - self.requeue.removed]
            self.requeue.removed = min_observer.next_item
        return current_item, item

    def __call__(self, mode=ReQueue.NEXT):
        self.mode = mode
        return self

    def __del__(self):
        with LockObject(self.requeue):
            self.requeue.observers.discard(self)
