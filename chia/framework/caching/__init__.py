_cache = dict()


def read_only_for_positional_args(fn):
    global _cache
    function_key = fn.__qualname__
    _cache[function_key] = dict()

    def wrapper(*args):
        global _cache
        args_key = args
        if args_key in _cache[function_key].keys():
            return _cache[function_key][args_key]
        else:
            value = fn(*args)
            _cache[function_key][args_key] = value
            return value

    return wrapper
