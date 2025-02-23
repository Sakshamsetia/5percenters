from flask import Flask, request, render_template, Response, send_file
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
import shutil
import main

app = Flask(__name__)

# Global input & output path variables
INPUT_FILE = ""
OUTPUT_VIDEO = ""

# Loading Default Page
@app.route("/")
def index():
    return render_template("index.html")


# Function that takes the output video and converts it into chunks of data, which can then be sent to browser
def getChunk(OUTPUT):
    cap = cv2.VideoCapture(OUTPUT)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield(b' --frame/r/n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
# Rendering graph
@app.route("/graph", methods=["GET", "POST"])
def graph():
    global OUTPUT_VIDEO
    # Saving file
    f = request.files.get("video")
    if not f:
        return "<h1> File Not Found </h1>"
    if not os.path.exists("./video_input/"):
        os.makedirs("./video_input/")
    INPUT_FILE = os.path.join("./video_input/", "22.mp4")
    f.save(INPUT_FILE)
    
    # Sending saved file to model
    OUTPUT_VIDEO = main.image(INPUT_FILE)
    return render_template("graph.html")

# Function that saves the file given by user, puts it into model, then uses getChunk() to convert into chunks, then sends it to browse
@app.route("/stream", methods=["GET", "POST"])
def stream():
    global OUTPUT_VIDEO
    # Converting it into chunks and then sending it to browser
    ch = getChunk(OUTPUT_VIDEO)
    return Response(ch, mimetype='multipart/x-mixed-replace; boundary=frame', status=200)

