class bar:
    def __init__(self, *iterable, append='', length=False, bar_size=100):
        """Inputs the same arguments as range or an iterable object."""
        if not iterable: raise ValueError()
        iterable = range(*iterable) if isinstance(iterable[0], int) else iterable[0]
        self.iterable, self.append, self.start, self.len, self.i, self.r, self.bar_size = iter(iterable), append, False, length if length else len(iterable), 0, None, bar_size

    def __iter__(self): return self

    def __next__(self):
        from time import monotonic
        now = monotonic()
        self.start = self.start or now
        took = int(now - self.start)
        if self.i >= self.len:
            print(f"\r{self.i: >6}/{self.len:<} (100%) \x1B[0;34m[\x1B[0;32m{'■' * self.bar_size}\x1B[0;34m]\x1B[0m  Took:" + (f'{took // 60: 3}m' if took >= 60 else '    ') + f'{took % 60:3}s  ' + self.append.format(self.r, self.i))
        self.r = next(self.iterable)
        eta = int((self.len - self.i) * (now - self.start) / self.i) if self.i else 0
        done = self.bar_size * self.i / self.len
        print('\r' + f"{self.i: 6}/{self.len:<} ({int(100 * self.i / self.len): 3}%) [\x1B[0;32m{{:·<{self.bar_size + 4}}}]  ".format('■' * int(done) + str(int(10 * (done % 1))) + '\x1B[0m')
              + ('ETA:' + (f'{eta // 60: 4}m' if eta >= 60 else '     ') + f'{eta % 60:3}s  ' if eta else '  ') + self.append.format(self.r, self.i), end='')
        self.i += 1
        return self.r
