import json

_current_context = None
_config_dict = {}


class ConfigurationContext:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        global _current_context
        self._parent_context: ConfigurationContext = _current_context
        _current_context = self

        if self._parent_context is None:
            self._full_description = self._description
        else:
            self._full_description = (
                f"{self._parent_context._description}.{self._description}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_context
        _current_context = self._parent_context

    def get(self, key, default_value=None):
        full_key = f"{self._full_description}.{key}"
        if full_key not in _config_dict.keys():
            _config_dict[full_key] = (default_value, False)

        return _config_dict[full_key][0]

    def set(self, key, value):
        full_key = f"{self._full_description}.{key}"

        if full_key not in _config_dict.keys():
            _config_dict[full_key] = (value, True)
        else:
            raise ValueError("Attempted to set key after first access.")


def get(key, default_value=None):
    if _current_context is not None:
        return _current_context.get(key, default_value)
    else:
        raise ValueError("Cannot get configuration entry without Configuration Context")


def set(key, value):
    if _current_context is not None:
        return _current_context.set(key, value)
    else:
        raise ValueError("Cannot set configuration entry without Configuration Context")


def get_config_dict():
    if _current_context is not None:
        return _current_context.get_config_dict()
    else:
        raise ValueError("Cannot get configuration dict without Configuration Context")


def clear():
    global _config_dict
    _config_dict = {}


def _dump_dict():
    dump_dict_default = {}
    dump_dict_custom = {}

    for key, value_pair in _config_dict.items():
        value, is_default = value_pair
        if is_default:
            dump_dict_default[key] = value
        else:
            dump_dict_custom[key] = value

    return dump_dict_default, dump_dict_custom


def dump_custom():
    return json.dumps(_dump_dict()[1], indent=2)


def dump_default():
    return json.dumps(_dump_dict()[0], indent=2)


# def save(path):
#     json.dump(open(path, "w"), _dump_dict())
#
#
# def load(path):
#     global _config_dict
#     assert len(_config_dict.items()) == 0
#
#     imported_dict = json.load(open(path))
#     for key, value in imported_dict.items():
#         set(key, value)


def main_context(func):
    def wrapper():
        with ConfigurationContext("global") as ctx:
            try:
                import sys
                import argparse

                parser = argparse.ArgumentParser()
                known_arguments, unknown_arguments = parser.parse_known_args()
                for unknown_argument in unknown_arguments:
                    # Look for config assignments
                    if "=" in unknown_argument:
                        # Parse assignment
                        config_key, value = unknown_argument.split("=", 2)
                        value = eval(value)
                        config_key_components = config_key.split(".")

                        # Create matching contexts
                        context_stack = []

                        for key_component in config_key_components[:-1]:
                            key_component_context = ConfigurationContext(key_component)
                            key_component_context.__enter__()
                            context_stack += [key_component_context]

                        # Set value
                        set(config_key_components[-1], value)

                        # Go back to usual context
                        for key_component_context in reversed(context_stack):
                            key_component_context.__exit__(None, None, None)

            except Exception as ex:
                print(f"Exception during configuration parsing: {ex}.")

            try:
                func()
            except Exception as ex:
                print(f"Unhandled exception during execution: {ex}.")

            print("Configuration dump (default):")
            print(dump_default())

            print("Configuration dump (custom):")
            print(dump_custom())

    return wrapper
