class FixedQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size 
        self.head = 0
        self.tail = 0
        self.count = 0 

    def push(self, item):
        if self.count < self.size:
            self.queue[self.tail] = item
            self.tail = (self.tail + 1) % self.size
            self.count += 1
        else:
            self.queue[self.tail] = item
            self.tail = (self.tail + 1) % self.size
            self.head = (self.head + 1) % self.size 

    def pop(self):
        if self.count == 0:
            return None
        item = self.queue[self.head]
        self.head = (self.head + 1) % self.size
        self.count -= 1
        return item

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        if key < 0 or key >= self.count:
            raise IndexError("Index out of range")
        return self.queue[(self.head + key) % self.size]

    def __iter__(self):
        idx = self.head
        num_items = self.count
        for _ in range(num_items):
            yield self.queue[idx]
            idx = (idx + 1) % self.size
            
    def __contains__(self, item):
        idx = self.head
        for _ in range(self.count):
            if self.queue[idx] == item:
                return True
            idx = (idx + 1) % self.size
        return False