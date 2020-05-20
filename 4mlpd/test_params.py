import flask
app = flask.Flask(__name__)
@app.route("/", methods=['POST'])
def index():
    params = flask.request.json
    return "\nargv = " + str(params[0]) + "\nbatch script params = " + str(params[1])
app.run(host='0.0.0.0', port=4567)

