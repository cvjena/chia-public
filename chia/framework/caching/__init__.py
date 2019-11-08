_cache = dict()


def read_only(fn):
    global _cache
    _cache[fn.__name__] = dict()

    def wrapper(*args):
        global _cache
        key = args
        if key in _cache[fn.__name__].keys():
            return _cache[fn.__name__][key]
        else:
            value = fn(*args)
            _cache[fn.__name__][key] = value
            return value

    return wrapper
