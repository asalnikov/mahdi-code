# Модуль для теста
import time
import random

ml_lib = "other"


def fit(data):
    time.sleep(15)
    l = len(data)
    if l > 0:
        #print(data[l // 2])
        return data[l // 2]
    else:
        print("need to fix db")
        exit(1)


def predict(ml, params):
    # task(n/2).(time_end - time_start) + len(argv[argc / 2 + 1])
    #time.sleep(15)
    return ml['time_end'] - ml['time_start'] + len(params[0][len(params[0]) // 2])
