import time

from PIL import Image


class NetworkResistantImage:
    @staticmethod
    def open(filename):
        wait_interval = 0.5
        last_exception = None
        while wait_interval <= 128.0:
            try:
                image = Image.open(filename)
                return image

            except Exception as ex:
                print(f"Cannot open {filename}: {ex}. Waiting {wait_interval} seconds.")
                last_exception = ex
                time.sleep(wait_interval)
                wait_interval *= 2.0

        raise last_exception
