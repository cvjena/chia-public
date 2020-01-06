import math


def constant_schedule(step, initial_lr):
    return initial_lr


def sgdr_schedule(
    step,
    minimum_lr: float,
    maximum_lr: float,
    T_0: int,
    T_mult: int = 2,
    maximum_lr_decay: float = 1.0,
    warmup_steps: int = 0,
    warmup_lr: float = 0.1,
):
    if step < warmup_steps:
        return warmup_lr

    step_cur = step
    maximum_lr_cur = maximum_lr
    T_i = T_0
    while step_cur > T_i:
        step_cur -= T_i
        T_i *= T_mult
        maximum_lr_cur *= maximum_lr_decay

    T_cur = float(step_cur)
    T_i = float(T_i)

    maximum_lr_cur = max(maximum_lr_cur, minimum_lr)

    eta_t = minimum_lr + 0.5 * (maximum_lr_cur - minimum_lr) * (
        1.0 + math.cos(math.pi * (T_cur / T_i))
    )
    return eta_t


def exponential_schedule(step, initial_lr, end_lr, decay_factor, steps_per_decay):
    current_decay_exponent = float(step) / steps_per_decay
    current_decay_factor = math.pow(decay_factor, current_decay_exponent)
    current_lr = max(initial_lr * current_decay_factor, end_lr)
    return current_lr


def constant_fn(**kwargs):
    return lambda step: constant_schedule(step, **kwargs)


def sgdr_fn(**kwargs):
    return lambda step: sgdr_schedule(step, **kwargs)


def exponential_fn(**kwargs):
    return lambda step: exponential_schedule(step, **kwargs)


all_schedules = {
    "constant": constant_fn,
    "sgdr": sgdr_fn,
    "exponential": exponential_fn,
}


def deserialize(config):
    if config["name"].lower() in all_schedules.keys():
        return all_schedules[config["name"].lower()](**config["config"])
    else:
        raise ValueError("Unknown classifier: %s" % config["name"].lower())


def get(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"name": identifier, "config": {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Unknown schedule: %s" % str(identifier))
