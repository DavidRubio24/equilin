

class RequestQueue:
    def __init__(self, followed=None):
        self.more = True
        self.followers: set[RequestQueue] = set()
        if followed is not None:
            followed.add_follower(self)

    def append(self, item):
        with LockObject(self):
            for observer in self.followers:
                observer.append(item)

    def off(self):
        self.more = False
        with LockObject(self):
            for observer in self.followers:
                observer.more = False

    def __iter__(self):
        observer = RequestQueueNext()
        self.add_follower(observer)
        return observer

    def add_follower(self, follower):
        with LockObject(self):
            self.followers.add(follower)

    def remove_follower(self, follower):
        with LockObject(self):
            self.followers.discard(follower)


class RequestQueueNext(RequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = []
        self.prev = []

    def __iter__(self): return self

    def __next__(self):
        # This thread has already processed the previous item,
        # we can pass it to the followers.
        if self.prev:
            RequestQueue.append(self, self.prev.pop())

        # Wait for the next item.
        while not self.items:
            # Unless we are told to stop.
            if not self.more:
                # Tell the followers to stop.
                self.off()
                raise StopIteration()  # End of iterator.
            time.sleep(0)

        # Save the used item to pass it to the followers next time.
        self.prev = [self.items.pop(0)]
        return self.prev[0]

    def append(self, item):
        self.items.append(item)


class RequestQueueLast(RequestQueueNext):
    def append(self, item):
        self.items = [item]  # Replace the last item.
