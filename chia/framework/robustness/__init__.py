import time

from PIL import Image

from chia import configuration


class NetworkResistantImage:
    @staticmethod
    def open(filename):
        with configuration.ConfigurationContext("NetworkResistantImage"):
            cumulative_wait_max = configuration.get(
                "cumulative_wait_max", 2.0 * 60.0 * 60.0
            )
            wait_interval_initial = configuration.get("wait_interval_initial", 0.5)

        # Set initial state
        wait_interval = wait_interval_initial
        cumulative_wait = 0.0
        last_exception = None

        while cumulative_wait <= cumulative_wait_max:
            try:
                image = Image.open(filename)
                return image

            except Exception as ex:
                print(f"Cannot open {filename}: {ex}. Waiting {wait_interval} seconds.")
                last_exception = ex
                time.sleep(wait_interval)
                cumulative_wait += wait_interval
                wait_interval *= 2.0

        raise last_exception
