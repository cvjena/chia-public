_current_context = None
_config_dict = {}


class ConfigurationContext:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        global _current_context
        self._parent_context = _current_context
        _current_context = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_context
        _current_context = self._parent_context

    def get_config_dict(self):
        global _config_dict
        if self._parent_context is not None:
            return self._parent_context.get(self._description, {})
        else:
            # I am the global context
            return _config_dict

    def get(self, key, default_value=None):
        config_dict = self.get_config_dict()

        if key not in config_dict.keys():
            config_dict[key] = default_value

        return config_dict[key]


def get(key, default_value=None):
    if _current_context is not None:
        return _current_context.get(key, default_value)
    else:
        raise ValueError("Cannot get configuration entry without Configuration Context")


def get_config_dict():
    if _current_context is not None:
        return _current_context.get_config_dict()
    else:
        raise ValueError("Cannot get configuration dict without Configuration Context")
