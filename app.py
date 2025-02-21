from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    render_template("index.html")


# add enctype="multipart/form-data" on form attribute
@app.route("/result", method=["GET", "POST"])
def result():

    # Getting file
    f = request.files["video"]

    # Giving request to model for data, and getting it as a dict
    # TODO
    data = {}
    
    # returning JSON output (variable - data)
    dataJSON = jsonify(data)
    render_template("result.html", data=data)

