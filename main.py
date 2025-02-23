from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
import shutil

def coincide(car_1,car_2,person_1,person_2):
    car_min_x, car_max_x = min(car_1[0],car_2[0]), max(car_1[0],car_2[0])
    person_min_x, person_max_x = min(person_1[0],person_2[0]), max(person_1[0],person_2[0])
    car_min_y , car_max_y = min(car_1[1],car_2[1]), max(car_1[1],car_2[1])
    person_min_y, person_max_y = min(person_1[1],person_2[1]), max(person_1[1],person_2[1])

    if car_max_x < person_min_x or person_max_x < car_min_x:
        return False

    if car_max_y < person_min_y or person_max_y < car_min_y:
        return False

    return True

def image(VIDEO_PATH):
    
    chair1 = None
    chair2 = None
    person1 = None
    person2 = None

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
    model = YOLO("yolov8n.pt")
    while True:
        ped_vio = False
        ret, frame = cap.read()
        
        if not ret:
            break
        height, width = frame.shape[:2]
        if height>=720 or width >= 1280:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if  True:
            
        # Load YOLOv8 model
            
            results = model.predict(frame)
            image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            for result in results:
                for box in result.boxes:
                # Boxes around images
                    confidence = box.conf[0]  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    cls = int(box.cls[0])  
                    label = model.names[cls] 
                    if label == "car":
                            chair1 = (x1,y1)
                            chair2 = (x2,y2)
                    elif label == "person":
                        person1 = (x1,y1)
                        person2 = (x2,y2) 
                        if chair2 is not None and chair1 is not None:    
                            if coincide(chair1,chair2 ,person1 , person2  ):
                                ped_vio = True
                            else:
                                ped_vio = False
                        else:
                            ped_vio = False
                    if confidence >0.5:
                        
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image_rgb, f'{label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
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
                    if j>0.75:
                        if i not in final_dict.keys():
                            final_dict[i] = 1
                        else:
                            final_dict[i] +=1
                        
            if ped_vio:
                cv2.putText(image_rgb, f"!!! Pedestrian Alert !!!", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 255, 20), 4)
            if "person" in final_dict and any(obj in final_dict for obj in ["car", "bus", "truck"]):
                violation = True
                
                
            else:
                violation = False
            if ((frame_number/fps) - prev_sec)>= 0.3:
                timeFrame.append({
                "Time (s)": frame_number/fps,
                "Pedestrians": final_dict.get("person", 0),
                "Vehicles": sum(final_dict.get(obj, 0) for obj in ["car", "bus", "truck"]),
                "Violation Detected?": violation
                }) 
                prev_sec = frame_number/fps   
            #Image download
            
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 
            video_frame = image_bgr
            if flag:
                height, width, layers = frame.shape
                flag = False            
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI
                video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
                video.write(video_frame)
            else:
                video.write(video_frame)
                
              
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
    graph_path = "./static/graph.png"
    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
    plt.close() 
    return video_name