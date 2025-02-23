# 5percenters
<h1>KrackHack Group project</h1>

<h3>Krackhack Project</h3>
<p>Traffic Perception Model is a web-based application that includes functionalities like real-time streaming, graphical visualizations, and image processing. It is built using Flask for the backend and serves frontend pages using HTML, CSS, and JavaScript with the help of Yolov8 and python libraries like opencv for Image detection</p>

<h3>Installation & Running the Project</h3>
<ol>
<li>
    Clone the Repository:

git clone https://github.com/Sakshamsetia/5percenters.git  
cd 5percenters  
</li>
<li>
Install Dependencies:<br>
pip install -r requirements.txt
</li>
<li>
Use this to open the webpage and click the link:-

python -m flask run
</li>
<li>
Upload video on webpage for Results
</li>
</ol>

<h3>Repository Structure</h3>
5percenters/  
│── static/                # Contains <br>static files (CSS, JS, Images)  <br>
│   ├── img1.jpg           # Sample image  <br>
│   ├── script.js          # JavaScript file  <br>
│   ├── style.css          # Stylesheet for UI  <br>
│   ├── style2.css         #  styles for graph page<br> 
│── templates/             # HTML templates  <br>
│   ├── index.html         # Main frontend page  <br>
│   ├── graph.html         # Page for displaying graphs  <br>
│   ├── stream.html        # Streaming interface  <br>
│── __pycache__/    <br>       # Compiled Python files  <br>
│── app.py                # Main Flask application  <br>
│── main.py                # Main DL Code <br>
│── yolov8n.pt             # Model file for YOLOv8  <br>
│── README.md              # Project documentation (this file)  <br>
│── requirements.txt       # Required Python packages  <br>

<h3>Code Explanation</h3>


This script processes a video using YOLOv8 to detect pedestrians and vehicles, checking for potential violations where pedestrians are too close to moving vehicles. It iterates through each frame, detects objects using a pre-trained YOLOv8 model, and uses bounding box overlap logic to determine if a pedestrian is in danger. If a violation is detected, it draws bounding boxes around detected objects and adds an alert message. The processed frames are compiled into a new output video, and a graph is generated to visualize violations over time.

<h3>Output</h3>

Processed video: Stored in video_output/output_video.mp4.

Violation Graph: Displays the frequency of violations over time.

