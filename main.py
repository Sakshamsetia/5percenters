from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
import shutil

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

folder_path = "C:/Users/Saksham Setia/Documents/Video"
shutil.rmtree(folder_path)  
os.makedirs(folder_path)  # Ensure the folder exists

counter = 1
# Path to the video file
video_path = 'C:/Users/Saksham Setia/Downloads/22.mp4'

# Directory to save frames
output_dir = 'frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

frame_number = 0
second_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_number % (int(fps)//4) == 0:
        
    # Load YOLOv8 model
        model = YOLO("yolov8n.pt")
        results = model.predict(frame)
        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
             # Boxes around images
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                confidence = box.conf[0]  
                cls = int(box.cls[0])  
                label = model.names[cls] 

                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_rgb, f'{label} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #Image download
        output_dir = 'C:/Users/Saksham Setia/Documents/Video'
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  
        cv2.imwrite(os.path.join(output_dir, f"{counter}.jpg"), image_bgr)

        
        counter+=1
        
        # Extract the first result
        result = results[0]

        # Get detected class IDs and corresponding names
        class_ids = result.boxes.cls.tolist()  # List of class IDs
        confidences = result.boxes.conf.tolist()  # Confidence scores
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]  # Convert IDs to names

        # Store the detected objects in a dictionary
        detected_objects = {}
        for name, conf in zip(class_names, confidences):
            if name in detected_objects:
                detected_objects[name].append(conf)  # Store confidence scores if multiple detections
            else:
                detected_objects[name] = [conf]
        # Print the dictionary
        
        final_dict = {}
        for i in detected_objects.keys():
            for j in detected_objects[i]:
                if j>=0.5:
                    if i not in final_dict.keys():
                        final_dict[i] = 1
                    else:
                        final_dict[i] +=1
                    
        print(final_dict)
        second_count += 1
    
    frame_number += 1

# Release the video capture object
cap.release()
#cv2.destroyAllWindows()

# Path to the folder containing images
image_folder = 'C:/Users/Saksham Setia/Documents/Video'
video_name = 'C:/Users/Saksham Setia/Documents/Video/output_video.mp4'  # Output video file name

# Get a sorted list of image files (assuming images are named sequentially)
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort(key=natural_sort_key)

# Read the first image to get dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI
fps = (fps//3)  # Frames per second
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through images and write to video
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)  # Add frame to the video

# Release the video writer and close
video.release()
print("Video has been successfully created!")