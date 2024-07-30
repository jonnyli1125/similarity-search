from time import perf_counter


def latency(fn, *args, **kwargs):
    start = perf_counter()
    value = fn(*args, **kwargs)
    return value, (perf_counter() - start)
