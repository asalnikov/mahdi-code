import threading
import time
import configparser
import signal
import flask
import importlib
import pickle
import tensorflow

# Configuration
conf_path = 'mlpd.conf'
sigfitupdate = signal.SIGUSR1
sigpredictupdate = signal.SIGUSR2
fit_time_file_path = '/tmp/mpld_fit_time'
commenting = True

def msg(s):
    if commenting:
        print(s)

import slurmdb_import as sdbi

class FitTime:
    @staticmethod
    def read():
        msg("Read fit time")
        fit_time = 0
        try:
            fit_time_file = open(fit_time_file_path, 'r')
            try:
                fit_time = float(fit_time_file.read())
            except Exception:
                pass
            fit_time_file.close()
        except Exception:
            pass
        return fit_time

    @staticmethod
    def write():
        msg("Write fit time")
        fit_time_file = open(fit_time_file_path, 'w')
        fit_time_file.write(str(time.time()))
        fit_time_file.close()

class ModelWrapper:
    class ModelNullException(Exception):
        pass

    def __init__(self, MLModule, MLModelFileStr):
        self.MLModelStable = 0
        self.MLModelStable_mtx = threading.Lock()
        self.MLModelFile_mtx = threading.Lock()
        self.MLModule = MLModule
        self.MLModelFileStr = MLModelFileStr
        self.fit_mtx = threading.Lock()

    def __change(self, MLModelNew):
        self.MLModelStable_mtx.acquire()
        self.MLModelStable = MLModelNew
        self.MLModelStable_mtx.release()

    def __save(self, MLModel):
        self.MLModelFile_mtx.acquire()
        msg("Save model")
        if self.MLModule.ml_lib == "tensorflow":
            MLModel.save(self.MLModelFileStr)
        elif self.MLModule.ml_lib == "other":
            with open(self.MLModelFileStr, 'wb') as f:
                pickle.dump(MLModel, f)
        else:
            exit(1)
        self.MLModelFile_mtx.release()

    def __load(self):
        MLModel = 0
        self.MLModelFile_mtx.acquire()
        if self.MLModule.ml_lib == "tensorflow":
            MLModel = tensorflow.keras.models.load_model(self.MLModelFileStr)  # need numpy==1.16.4
        elif self.MLModule.ml_lib == "other":
            with open(self.MLModelFileStr, 'rb') as f:
                MLModel = pickle.load(f)
        else:
            exit(1)
        self.MLModelFile_mtx.release()
        return MLModel

    def load(self):
        msg("Load Model")
        self.__change(self.__load())

    def predict(self, params):
        msg("Predict time")
        if self.MLModelStable == 0:
            raise self.ModelNullException
        self.MLModelStable_mtx.acquire()
        try:
            ans = self.MLModule.predict(self.MLModelStable, params)
        except Exception:
            raise
        finally:
            self.MLModelStable_mtx.release()
        return ans

    def fit(self):
        self.fit_mtx.acquire()
        msg('Fit the model')
        MLModelNew = self.MLModule.fit(sdbi.slurm_db())
        self.__change(MLModelNew)
        self.__save(MLModelNew)
        FitTime.write()
        self.fit_mtx.release()

    def fit_is_worked(self):
        return self.fit_mtx.locked()

class FitSystem:
    def __init__(self, model, FitUpdateTime):
        self.__FitUpdateTime = FitUpdateTime
        self.__model = model
        self.__fit_thread = threading.Thread(target=self.__fit_thread_func, daemon=True)

    def start(self):
        msg("Start fit thread")
        self.__fit_thread.start()
        self.__SignalHandlerStart()

    def __fit_thread_func(self):
        load_model_flag = True
        while True:
            time_to_fit_update = time.time() - FitTime.read()
            diff_time = time_to_fit_update - self.__FitUpdateTime
            if diff_time < 0:
                if load_model_flag:
                    self.__model.load()
                    load_model_flag = False
                self.__sleep(-diff_time)
            self.__model.fit()
            self.__sleep(self.__FitUpdateTime)

    def __SignalHandlerStart(self):
        def sigfitupdate_handler(signum, frame):
            msg("Caught the signal to force re-fit the model.")
            if self.__model.fit_is_worked():
                return 0
            sigfitupdate_thread = threading.Thread(target=self.__model.fit, daemon=True)
            sigfitupdate_thread.start()
            return 0
        signal.signal(sigfitupdate, sigfitupdate_handler)

    @staticmethod
    def __sleep(sec):
        msg('Sleep ' + str(int(sec)) + ' seconds')
        time.sleep(sec)

class PredictSystem:
    def __init__(self, model, ServerHost, ServerPort):
        self.ServerHost = ServerHost
        self.ServerPort = ServerPort
        self.model = model

    def start(self):
        msg("Start predict server")
        self.__SignalHandlerStart()
        self.__ServerStart()

    def __SignalHandlerStart(self):
        def sigpredictupdate_handler(signum, frame):
            msg("Caught the signal to force load the model.")
            self.model.update()
        signal.signal(sigpredictupdate, sigpredictupdate_handler)

    def __ServerStart(self):
        app = flask.Flask("predict server")
        @app.route("/", methods=['POST'])
        def index():
            params = flask.request.json
            try:
                ans_time = self.model.predict(params)
                return str(time.strftime('%H:%M:%S', time.gmtime(ans_time)))
            except ModelWrapper.ModelNullException:
                return 'Нет результата. Модель еще не доготовилась!'
            except Exception:
                return 'Ошибка в работе функции predict()'
        app.run(host=self.ServerHost, port=self.ServerPort)

class Configuration:
    def __init__(self):
        msg("Read configuration file")
        conf = configparser.ConfigParser()
        conf.read(conf_path)

        msg("Load ML Module")
        self.MLModule = importlib.import_module(conf.get("MLPD", "MLModule"))

        self.FitUpdateTime = int(conf.get("MLPD", "FitUpdateTime")) * 60 * 60
        self.MLModelFileStr = conf.get("MLPD", "SavedModel")
        self.ServerHost = conf.get("MLPD", "ServerHost")
        self.ServerPort = conf.get("MLPD", "ServerPort")

def mlpd():
    msg("Initialization predict server")
    cfg = Configuration()

    model = ModelWrapper(cfg.MLModule, cfg.MLModelFileStr)

    fitSystem = FitSystem(model, cfg.FitUpdateTime)
    fitSystem.start()

    predictSystem = PredictSystem(model, cfg.ServerHost, cfg.ServerPort)
    predictSystem.start()

    return 0

if __name__ == "__main__":
    mlpd()
