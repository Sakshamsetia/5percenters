from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    render_template("index.html")


