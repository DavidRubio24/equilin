from __future__ import annotations

import itertools
import time

from utils.lockobject import LockObject


class RequestQueue:
    def __init__(self, followed: RequestQueue = None):
        self.more = True
        self.followers: set[RequestQueue] = set()
        if followed is not None:
            followed.add_follower(self)
            self.more = followed.more

    def append(self, item):
        for observer in list(self.followers):
            observer.append(item)

    def off(self):
        self.more = False
        for observer in list(self.followers):
            observer.more = False

    def __iter__(self) -> RequestQueueNext: return RequestQueueNext(self)

    def __call__(self, type_: type): return type_(self)

    def add_follower(self, *followers):
        with LockObject(self):
            self.followers.update(followers)

    # TODO: get rid of this function.
    def remove_follower(self, *followers):
        with LockObject(self):
            self.followers.difference_update(followers)


class RequestQueueNext(RequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = []
        self.prev = []

    def __iter__(self): return self

    def __next__(self):
        # The previous item has already been processed, pass it to the followers.
        if self.prev:
            RequestQueue.append(self, self.prev.pop())

        # Wait for the next item.
        while not self.items:
            # Unless we are told to stop.
            if not self.more:
                # Tell the followers to stop.
                self.off()
                self.append = lambda x: None  # TODO: will this work?
                self.prev = []
                raise StopIteration()  # End of iterator.
            time.sleep(0)

        # Save the used item to pass it to the followers next time.
        self.prev = [self.items.pop(0)]
        return self.prev[0]

    def append(self, item): self.items.append(item)

    def has_next(self) -> bool: return bool(self.items)


class RequestQueueLast(RequestQueueNext):
    def append(self, item): self.items = [item]  # Replace the last item.


def join_queues(queues: list[RequestQueue], lenghts: list[int], joined: RequestQueue):
    """Joins two queues together asuming the first element of each item is a corresponding increasing id."""
    iterators = [queue.__iter__() for queue in queues]

    # While one queue has more items:
    while any(it.has_next() for it in iterators):
        # Get all the next items that are inmediatly available.
        # For the ones that are not available create a fake item one with an infinite id.
        have_next = [it.__next__() if it.has_next() else [float('inf')] for it, in iterators]
        while [x for x in have_next if x[0] != float('inf')]:
            min_ = min(have_next, key=lambda x: x[0])
            results = [result[1:] if result[0] == min_ else [None] * lenghts[idx] for idx, result in enumerate(have_next)]
            joined.append((min_[0], *results))
            have_next = [x if x is not None and x[0] >= min_ else [float('inf')] for x in have_next]


def join_queues_waiting(joined: RequestQueue, *queues: RequestQueue):
    """Joins two queues together asuming the first element of each item is a corresponding increasing id."""
    iterators = [queue.__iter__() for queue in queues]

    items = [None] * len(queues)

    # While there are more items in the queues (has_next()) or there will be more items (more):
    while any(it.has_next() or it.more for it in iterators):
        # Get at least one item for each queue (only if we didn't have one already).
        items = [item or it.__next__() for it, item in zip(iterators, items)]
        # Find the minimum id.
        min_id = min(map(lambda x: x[0] if x is not None else float('inf'), items))
        # Add the items with the minimum id to the joined queue.
        minims = [item[1:] if item[0] == min_id else [None] * len(item[1:]) for item in items]
        joined.append((min_id, *tuple(itertools.chain(*minims))))
        # Remove the items with the minimum id from the queues.
        items = [item if item is not None and item[0] != min_id else None for item in items]
