import datetime
import time
fit_time = 0
fit_time_file_path = '/tmp/mpld_fit_time'

fit_time_file = open(fit_time_file_path, 'w')
q = time.time()
print(q)
fit_time_file.write(str(q))


try:
    fit_time_file = open(fit_time_file_path, 'r')
    w = fit_time_file.read()
    w = float(w)
    print(w)
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(w))
    print(s)
    fit_time_file.close()
except IOError:
    print("IOError")
except Exception:
    print("Exception")
print("hello")


