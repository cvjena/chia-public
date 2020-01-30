import os

DESCRIPTION = "Tensorflow Log Level Setting"


def startup_fn():
    # Try turning off tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return True
