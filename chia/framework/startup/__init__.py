from chia.framework.startup import gpu_detection, tensorflow_log_level, user_config

_startup_modules = [user_config, gpu_detection, tensorflow_log_level]


def startup():
    print("This is CHIA: Concept Hierarchies for Incremental and Active Learning")

    for startup_module in _startup_modules:
        print(f"Running startup module: {startup_module.DESCRIPTION}")
        retval = startup_module.startup_fn()
        if not retval:
            print("Startup failed. Quitting.")
            quit(-1)
