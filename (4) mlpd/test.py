import flask
app = flask.Flask(__name__)
@app.route("/", methods=['GET'])
def index():
    params = flask.request.json
    return str(456456)
app.run(host='0.0.0.0', port=4567)