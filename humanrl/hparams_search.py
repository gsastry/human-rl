from collections import namedtuple
import random

import numpy as np


LinearRange = namedtuple("LinearRange", ["start", "end"])
LogRange = namedtuple("LogRange", ["start", "end"])


ranges = {
  "learning_rate": LogRange(0.000001, 0.001),
  "exploration_fraction": LinearRange(0.1, 1.0),
}

# todo - maybe adam epsilon (default 1e-4 in this implementation, but there's a tensorflow note saying that
# they've noticed different values being better for different tasks)


def sample_hparams():
    hparams = {}
    for k, sample_range in ranges.items():
        if isinstance(sample_range, (LogRange, LinearRange)):
            if isinstance(sample_range[0], int):
                # LogRange not valid for ints
                hparams[k] = random.randint(sample_range[0], sample_range[1])
            elif isinstance(sample_range[0], float):
                start, end = sample_range
                if isinstance(sample_range, LogRange):
                    start, end = np.log10(start), np.log10(end)

                choice = np.random.uniform(start, end)
                if isinstance(sample_range, LogRange):
                    choice = np.exp(choice)
                hparams[k] = choice
    return hparams

if __name__ == "__main__":
    hparams = sample_hparams()
    for hp, value in hparams.items():
        print("--{}={}".format(hp, value))