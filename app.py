from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    render_template("index.html")


# add enctype="multipart/form-data" on form attribute
@app.route("/result")
def result():

    # Getting file
    f = request.files["video"]
    f.save("E:\\KrackHack\\")

    # Running model function (TODO)
    # Data is return value
    data = {}
    
    # returning JSON output (variable - data)
    dataJSON = jsonify(data)
    return dataJSON

