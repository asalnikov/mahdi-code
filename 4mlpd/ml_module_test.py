# Модуль для теста
import time
import random

ml_lib = "other"

def fit(logs, sinfo):
    time.sleep(60)
    return [[1, 2, 3, 4, 5], "qwerty", [0.5, 1.8, 19, 7, 4398]]

def predict(ml, params):
    time.sleep(5)
    return random.choice(ml[2])


