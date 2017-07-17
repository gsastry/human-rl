import random
from functools import wraps
from time import time


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def shuffle(lst):
    random.shuffle(lst)
    return lst


def timing(f):
    """https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator#15136422
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


ACTION_MEANING = {
    None: "NONE",
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_MEANING_TO_ACTION = {v: k for k, v in ACTION_MEANING.items()}

ACTION_SETS = {
    "Pong": [0, 1, 3, 4, 11, 12],
    "SpaceInvaders": [0, 1, 3, 4, 11, 12],
}

SAFE_ACTION_MAPPINGS = {"SpaceInvaders": {0: 0, 1: 0, 2: 2, 3: 3, 4: 2, 5: 3}}
