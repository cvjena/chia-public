class Sample:
    def __init__(self, source=None, data=None, history=None, uid=None):
        if history is not None:
            self.history = history
        elif source is not None:
            self.history = [('init', '', source)]
        else:
            self.history = []

        if data is not None:
            self.data = data
        else:
            if uid is not None:
                self.data = {'uid': uid}
            else:
                raise ValueError('Need UID for sample!')

    def add_resource(self, source, resource_id, datum):
        assert resource_id not in self.data.keys()

        new_history = self.history + [('add', resource_id, source)]
        new_data = {resource_id: datum, **self.data}

        return Sample(data=new_data, history=new_history)

    def apply_on_resource(self, source, resource_id, fn):
        assert resource_id in self.data.keys()
        new_history = self.history + [('apply', resource_id, source)]
        new_data = {k: v if k != resource_id else fn(v) for k, v in self.data.items()}

        return Sample(data=new_data, history=new_history)

    def get_resource(self, resource_id):
        return self.data[resource_id]

    def __eq__(self, other):
        return self.data['uid'] == other.data['uid']
