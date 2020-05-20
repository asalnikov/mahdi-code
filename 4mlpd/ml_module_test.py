# Модуль для теста
import time
import random

ml_lib = "other"


def fit(data):
    l = len(data)
    if l > 0:
        return data[l // 2]
    else:
        print("need to fix db")
        exit(1)


def predict(ml, params):
    # task(n/2).(time_end - time_start) + len(argv[argc / 2 + 1])
    return ml[46 - 8] - ml[45 - 8] + len(params[0][len(params[0]) // 2])
