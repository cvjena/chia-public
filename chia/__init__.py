import os
from chia.framework import configuration

print("This is CHIA: Concept Hierarchies for Incremental and Active Learning")

# Try turning off tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# GPU stuff
try:
    from GPUtil import GPUtil
    import math

    gpus = GPUtil.getGPUs()
    assert len(gpus) == 1, "Only one GPU is supported right now."
    gpu = gpus[0]
    configuration.set_system("gpu0_vram", math.trunc(gpu.memoryTotal / 102.4) / 10)
except Exception as ex:
    print(f"Could not read available VRAM: {str(ex)}. Setting default value of 4GiB.")
    configuration.set_system("gpu0_vram", 4.0)

print(f"GPU0: {configuration.get_system('gpu0_vram')} Gib VRAM")
