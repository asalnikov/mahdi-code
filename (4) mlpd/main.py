import os
import threading
import time
import configparser
import signal
import flask
import importlib
import sklearn
import tensorflow
import slurmdb_import as sdbi


# Configuration
conf_path = "mlpd.conf"
sigfitupdate = signal.SIGUSR1
sigpredictupdate = signal.SIGUSR2


# Read configuration file
conf = configparser.ConfigParser()
conf.read(conf_path)
MLModule = conf.get("MLPD", "MLModule")
MLModule = importlib.import_module(MLModule)
FitUpdateTime = conf.get("MLPD", "FitUpdateTime")
FitUpdateTime = int(FitUpdateTime) * 60 * 60
MLModelFile = conf.get("MLPD", "SavedModel")


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
    if MLModule.ml_lib == "tensorflow":
        MLModel.save(MLModelFile)
    elif MLModule.ml_lib == "sklearn":
        exit(1)
    else:
        exit(1)
    MLModelFile_mtx.release()

def load_model():
    MLModel = 0
    MLModelFile_mtx.acquire()
    if MLModule.ml_lib == "tensorflow":
        MLModel = tensorflow.keras.models.load_model(MLModelFile)
    elif MLModule.ml_lib == "sklearn":
        exit(1)
    else:
        exit(1)
    MLModelFile_mtx.release()
    return MLModel


# Fit thread (1, 2)
fit_mtx = threading.Lock()

def fit_unblocked():
    # 4.4.2.1 - 4.4.2.4
    MLModelNew = MLModule.fit(sdbi.logs(), sdbi.sinfo())
    MLModelStable_change(MLModelNew)
    save_model(MLModelNew)

def fit():
    fit_mtx.acquire()
    fit_unblocked()
    fit_mtx.release()

def fit_thread_func():
    while True:
        fit()
        time.sleep(FitUpdateTime)
fit_thread = threading.Thread(target=fit_thread_func)
fit_thread.start()

def sigfitupdate_handler(signum, frame):
    sigfitupdate_thread = threading.Thread(target=fit)
    sigfitupdate_thread.start()
signal.signal(sigfitupdate, sigfitupdate_handler)


# Predict thread (0)
def sigpredictupdate_handler(signum, frame):
    MLModelNew = load_model()
    MLModelStable_change(MLModelNew)
signal.signal(sigpredictupdate, sigpredictupdate_handler)

def predict(params):
    return MLModule.predict(MLModelStable, params)

app = flask.Flask("predict server")
@app.route("/", methods=['POST'])
def index():
    if MLModelStable == 0:
        return "Модель еще не доготовилась!"
    else:
        params = flask.request.json
        return str(time.strftime('%H:%M:%S', time.gmtime(predict(params))))
app.run(host='0.0.0.0', port=4567)





