import numpy as np


def probtoprob(rate, a=1, b=12):
    return 1 - (1 - rate) ** (a / b)
