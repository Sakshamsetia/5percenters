from flask import Flask, request, render_template, Response
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
import shutil
import main

app = Flask(__name__)

# Global input & output path variables
INPUT_PATH = "./video_input/"
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
        



# Function that saves the file given by user, puts it into model, then uses getChunk() to convert into chunks, then sends it to browse
@app.route("/stream", methods=["GET", "POST"])
def stream():

    # Saving file
    f = request.files.get("video")
    if not f:
        return "<h1> No file uploaded </h1>"
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
    inputFile = os.path.join(INPUT_PATH, "22.mp4")
    f.save(inputFile)
    
    # Sending saved file to model
    OUTPUT_VIDEO = main.image(inputFile)

    # Converting it into chunks and then sending it to browser
    ch = getChunk(OUTPUT_VIDEO)
    return Response(ch, mimetype='multipart/x-mixed-replace; boundary=frame', status=200)



