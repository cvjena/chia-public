class Sample:
    def __init__(self, source):
        self.data = dict()
        self.history = [('init', '', source)]

    def add_resource(self, source, id, datum):
        assert id not in self.data.keys
        self.history += [('add', id, source)]
        self.data[id] = datum

    def apply_on_resource(self, source, id, fn):
        assert id in self.data.keys
        self.history += [('apply', id, source)]
        self.data[id] = fn(self.data[id])

    def get_resource(self, id):
        return self.data[id]