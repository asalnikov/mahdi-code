import threading
import time
import configparser
import signal
import flask
import importlib
import pickle
import tensorflow


# Configuration
conf_path = "mlpd.conf"
sigfitupdate = signal.SIGUSR1
sigpredictupdate = signal.SIGUSR2
commenting = True
fit_time_file_path = '/tmp/mpld_fit_time'


def msg(s):
    if commenting:
        print(s)

msg("Initialization predict server")

import slurmdb_import as sdbi

# Configuration
msg("Read configuration file")
conf = configparser.ConfigParser()
conf.read(conf_path)
MLModule = conf.get("MLPD", "MLModule")
msg("Load ML Module")
MLModule = importlib.import_module(MLModule)
FitUpdateTime = conf.get("MLPD", "FitUpdateTime")
FitUpdateTime = int(FitUpdateTime) * 60 * 60
MLModelFileStr = conf.get("MLPD", "SavedModel")
ServerHost = conf.get("MLPD", "ServerHost")
ServerPort = conf.get("MLPD", "ServerPort")

# ML
MLModelStable = 0
MLModelStable_mtx = threading.Lock()
def MLModelStable_change(MLModelNew):
    MLModelStable_mtx.acquire()
    global MLModelStable
    MLModelStable = MLModelNew
    MLModelStable_mtx.release()

MLModelFile_mtx = threading.Lock()

def save_model(MLModel):
    MLModelFile_mtx.acquire()
    msg("Save model")
    if MLModule.ml_lib == "tensorflow":
       MLModel.save(MLModelFileStr)
       exit(1)
    elif MLModule.ml_lib == "other":
        with open(MLModelFileStr, 'wb') as f:
            pickle.dump(MLModel, f)
    else:
        exit(1)
    MLModelFile_mtx.release()

def load_model():
    MLModel = 0
    MLModelFile_mtx.acquire()
    msg("Load model")
    if MLModule.ml_lib == "tensorflow":
        MLModel = tensorflow.keras.models.load_model(MLModelFileStr) # need numpy==1.16.4
    elif MLModule.ml_lib == "other":
        with open(MLModelFileStr, 'rb') as f:
            MLModel = pickle.load(f)
    else:
        exit(1)
    MLModelFile_mtx.release()
    return MLModel

def update_model():
    MLModelNew = load_model()
    MLModelStable_change(MLModelNew)


# Fit thread (1, 2)
msg("Start fit thread")
fit_mtx = threading.Lock()

def read_fit_time():
    msg("Read fit time")
    fit_time = 0
    try:
        fit_time_file = open(fit_time_file_path, 'r')
        fit_time = float(fit_time_file.read())
        fit_time_file.close()
    except Exception:
        pass
    return fit_time

def write_fit_time():
    msg("Write fit time")
    fit_time_file = open(fit_time_file_path, 'w')
    fit_time_file.write(str(time.time()))
    fit_time_file.close()

def fit():
    fit_mtx.acquire()
    MLModelNew = MLModule.fit(sdbi.slurm_db())
    MLModelStable_change(MLModelNew)
    save_model(MLModelNew)
    write_fit_time()
    fit_mtx.release()

def fit_thread_func():
    update_model_flag = True
    while True:
        time_to_fit_update = time.time() - read_fit_time()
        diff_time = time_to_fit_update - FitUpdateTime
        if diff_time < 0:
            if update_model_flag:
                update_model()
                update_model_flag = False
            time.sleep(-diff_time)
        fit()
        time.sleep(FitUpdateTime)
fit_thread = threading.Thread(target=fit_thread_func, daemon=True)
fit_thread.start()

def sigfitupdate_handler(signum, frame):
    msg("Caught the signal to force re-fit the model.")
    if fit_mtx.locked():
        return 0
    sigfitupdate_thread = threading.Thread(target=fit, daemon=True)
    sigfitupdate_thread.start()
    return 0
signal.signal(sigfitupdate, sigfitupdate_handler)


# Predict thread (0)
msg("Start predict server")
def sigpredictupdate_handler(signum, frame):
    msg("Caught the signal to force load the model.")
    update_model()
signal.signal(sigpredictupdate, sigpredictupdate_handler)

def predict(params):
    MLModelStable_mtx.acquire()
    ans = MLModule.predict(MLModelStable, params)
    MLModelStable_mtx.release()
    return ans

app = flask.Flask("predict server")
@app.route("/", methods=['POST'])
def index():
    if MLModelStable == 0:
        return "Нет результата. Модель еще не доготовилась!"
    else:
        params = flask.request.json
        msg(str(params))
        return str(time.strftime('%H:%M:%S', time.gmtime(predict(params))))
app.run(host=ServerHost, port=ServerPort)
