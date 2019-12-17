import json
import os

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

    def get(self, key, default_value=None, no_default=False):
        assert "." not in key

        full_key = f"{self._full_description}.{key}"
        if full_key not in _config_dict.keys():
            if no_default:
                raise ValueError(
                    f"Requested read of config key {full_key} for which there is no default given!"
                )
            else:
                _config_dict[full_key] = {
                    "value": default_value,
                    "is_default": True,
                    "access_ctr": 0,
                }
        else:
            if (
                _config_dict[full_key]["value"] == default_value
                and not _config_dict[full_key]["is_default"]
            ):
                print(
                    f"WARNING: Default value for {full_key} repeated in custom configuration!"
                )
            if (
                _config_dict[full_key]["value"] != default_value
                and _config_dict[full_key]["is_default"]
            ):
                print(
                    f"WARNING: Default value for {full_key} defined in different ways!"
                )

        value = _config_dict[full_key]["value"]
        _config_dict[full_key]["access_ctr"] += 1
        return value

    def set(self, key, value):
        assert "." not in key

        full_key = f"{self._full_description}.{key}"

        if full_key not in _config_dict.keys():
            _config_dict[full_key] = {
                "value": value,
                "is_default": False,
                "access_ctr": 0,
            }
        elif _config_dict[full_key]["access_ctr"] == 0:
            _config_dict[full_key]["value"] = value
            _config_dict[full_key]["is_default"] = False
        else:
            raise ValueError(f"Attempted to set key {full_key} after first access.")


def get_system(key):
    full_key = f"system.{key}"
    if full_key not in _config_dict.keys():
        raise ValueError(f"Unknown system config key {key}.")
    else:
        return _config_dict[full_key]


def set_system(key, value):
    full_key = f"system.{key}"
    if full_key in _config_dict.keys():
        raise ValueError(f"System config key {key} already exists.")
    else:
        _config_dict[full_key] = value


def get(key, default_value=None, no_default=False):
    if _current_context is not None:
        return _current_context.get(key, default_value, no_default)
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

    for key, value_dict in _config_dict.items():
        value = value_dict["value"]
        is_default = value_dict["is_default"]

        if is_default:
            dump_dict_default[key] = value
        else:
            dump_dict_custom[key] = value

    return dump_dict_default, dump_dict_custom


def dump_custom_json():
    return json.dumps(_dump_dict()[1], indent=2)


def dump_default_json():
    return json.dumps(_dump_dict()[0], indent=2)


def dump_custom_dict():
    return dict(_dump_dict()[1])


def dump_default_dict():
    return dict(_dump_dict()[0])


def _load_json(path):
    if len(_config_dict.items()) != 0:
        print(
            f"WARNING: Loading multiple JSON files for configuration. Current file: {path}"
        )
    else:
        print(f"Using config source: {path}")

    imported_dict = json.load(open(path))
    for key, value in imported_dict.items():
        _update(key, value)


def _update(config_key, value):
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


def main_context(config_sources=None):
    def inner_main_context(func):
        def wrapper():
            with ConfigurationContext("global") as ctx:
                if config_sources is None:
                    total_config_sources = ["configuration.json"]
                else:
                    total_config_sources = config_sources

                for config_source in total_config_sources:
                    try:
                        if os.path.exists(config_source):
                            _load_json(config_source)
                        else:
                            print(
                                f"WARNING: Skipping configuration file {config_source} that could not be found!"
                            )
                    except Exception as ex:
                        print(f"Exception during defaults loading: {ex}")
                        raise ex

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
                            try:
                                value = eval(value)
                            except (
                                SyntaxError,
                                NameError,
                                TypeError,
                                ZeroDivisionError,
                            ):
                                value = str(value)
                                print(
                                    f"WARNING: Interpreting {config_key}={value} as string!"
                                )
                            _update(config_key, value)
                        else:
                            if os.path.exists(unknown_argument):
                                _load_json(unknown_argument)
                            else:
                                raise ValueError(
                                    f"Could not process argument {unknown_argument}"
                                )

                except Exception as ex:
                    print(f"Exception during configuration parsing: {ex}.")
                    raise ex

                print("Configuration dump (custom):")
                print(dump_custom_json())
                try:
                    func()
                except Exception as ex:
                    the_ex = ex
                    print(f"Unhandled exception during execution: {ex}.")
                else:
                    the_ex = None

                print("Configuration dump (custom):")
                print(dump_custom_json())
                print("Configuration dump (default):")
                print(dump_default_json())

                # Check access counter
                for key, value_dict in _config_dict.items():
                    if value_dict["access_ctr"] == 0:
                        print(f"WARNING: configuration entry {key} unused.")

                if the_ex is not None:
                    raise the_ex

        return wrapper

    return inner_main_context
