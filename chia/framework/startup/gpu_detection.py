from chia.framework import configuration

DESCRIPTION = "GPU VRAM Detection"


def startup_fn():
    try:
        from GPUtil import GPUtil
        import math
        import os

        # Disable GPUS if desired
        if "CHIA_CPU_ONLY" in os.environ.keys():
            if os.environ["CHIA_CPU_ONLY"] == "1":
                print(
                    "Requested CPU only operation."
                    "Disabling all GPUS via environment variable."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                raise ValueError("Requested CPU-only operation.")

        gpus = GPUtil.getGPUs()
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            cuda_filter = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            print(f"Found CUDA device filter: {cuda_filter}")
            gpus = [gpu for gpu in gpus if str(gpu.id) in cuda_filter]

        available_gpus = []
        for gpu in gpus:
            print(f"Found GPU: {gpu.name}")
            if gpu.memoryUtil < 0.5:
                available_gpus += gpus
            else:
                print(
                    "Cannot use this GPU because of its "
                    f"memory utilization @ {int(100.0 * gpu.memoryUtil)}%."
                )

        if len(available_gpus) > 1:
            print("Only one GPU is supported right now.")
            return False

        if len(available_gpus) < 1:
            print("Need an available GPU!")
            return False

        gpu = available_gpus[0]
        configuration.set_system("gpu0_vram", math.trunc(gpu.memoryTotal / 102.4) / 10)

    except Exception as ex:
        print(
            f"Could not read available VRAM: {str(ex)}. Setting default value of 4GiB."
        )
        configuration.set_system("gpu0_vram", 4.0)

    print(f"GPU0: {configuration.get_system('gpu0_vram')} Gib VRAM")
    return True
