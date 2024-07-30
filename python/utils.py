import argparse
from time import perf_counter


def latency(fn, *args, **kwargs):
    start = perf_counter()
    value = fn(*args, **kwargs)
    return value, (perf_counter() - start)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    return parser
