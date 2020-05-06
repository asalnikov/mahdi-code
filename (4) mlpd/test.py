import flask
app = flask.Flask(__name__)
@app.route("/", methods=['POST'])
def index():
    params = flask.request.json
    return str(456456) + str(params) + "\n"
app.run(host='0.0.0.0', port=4567)
