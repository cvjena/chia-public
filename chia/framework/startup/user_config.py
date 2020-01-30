import os

from chia.framework import configuration

DESCRIPTION = "User Configuration"


def startup_fn():
    chiarc_filename = os.path.expanduser("~/.chiarc")
    if os.path.exists(chiarc_filename):
        print("Found .chiarc")
        import json

        try:
            with open(chiarc_filename) as chiarc_file:
                chiarc_content = json.load(chiarc_file)
                assert isinstance(chiarc_content, dict)
                for key, value in chiarc_content.items():
                    configuration.set_system(key, value)
        except Exception as ex:
            print(f"Could not process .chiarc: {ex}, {ex.args}")
            return False

    else:
        print("No .chiarc found")

    return True
