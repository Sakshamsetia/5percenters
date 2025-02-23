from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
import shutil

def image(VIDEO_PATH):
    flag = True
    timeFrame = []
    prev_sec = -0.1
    base_dir = os.getcwd()
    output_folder = os.path.join(base_dir, "video_output")
    if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    video_name = os.path.join(output_folder, "output_video.mp4")
    folder_path = output_folder
    shutil.rmtree(folder_path)  
    os.makedirs(folder_path)  # Ensure the folder exists

    # Path to the video file
    video_path = os.path.join(VIDEO_PATH)



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
        
        if True:
            
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            #Image download
        
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 
            video_frame = image_bgr
            if flag:
                height, width, layers = frame.shape
                flag = False            
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI
                fps = (fps)  # Frames per second
                video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
                video.write(video_frame)
            else:
                video.write(video_frame)
                
            
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
                        
            
            if "person" in final_dict and any(obj in final_dict for obj in ["car", "bus", "truck"]):
                violation = True
            else:
                violation = False
            if ((frame_number/fps) - prev_sec)>= 0.1:
                timeFrame.append({
                "Time (s)": frame_number/fps,
                "Pedestrians": final_dict.get("person", 0),
                "Vehicles": sum(final_dict.get(obj, 0) for obj in ["car", "bus", "truck"]),
                "Violation Detected?": violation
                }) 
                prev_sec = frame_number/fps     
            second_count += 1
        
        frame_number += 1
    video.release()
    print("Video has been successfully created!")
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    time_values = [entry["Time (s)"] for entry in timeFrame]
    computed_values = [2 * entry["Vehicles"] + entry["Pedestrians"] if entry["Violation Detected?"] else 0 for entry in timeFrame]
    # Create the bar graph
    plt.figure(figsize=(12, 6))
    plt.plot(time_values, computed_values, marker='o', linestyle='-', color='red', label="2*Vehicles + Pedestrians (if Violation)")

    # Fill the area under the curve
    plt.fill_between(time_values, computed_values, color='red', alpha=0.3)

    # Graph labels and title
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Violation Impact Score", fontsize=12)
    plt.title("Traffic Violation Impact Over Time", fontsize=14)
    plt.legend()
    plt.xticks([])  # Hide tick marks
    plt.gca().axes.get_xaxis().set_visible(False)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Show the plot
    graph_path = "violation_shaded_plot.png"
    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
    plt.close() 
    return video_name

