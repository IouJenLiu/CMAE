from heapq import heappush, heappop, nsmallest, heapify


class PQ(object):
    def __init__(self, data=None):
        if data is None:
            self.Q = []
        else:
            self.Q = data
            heapify(self.Q)

    def push(self, elem):
        heappush(self.Q, elem)

    def pop(self):
        return heappop(self.Q)

    def nsmallest(self, n):
        return nsmallest(n, self.Q)

    def __len__(self):
        return len(self.Q)