import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from codrone_edu.drone import Drone
import time

# Load TensorFlow model before drone starts
print("Loading TensorFlow model...")
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
print("Model loaded successfully!")

# Full COCO label set (ensure this matches your model's labels)
coco_labels = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
    8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter",
    15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant",
    23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "apple", 53: "banana", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "cell phone", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush", 100: "apple"
}

# Open Mac camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access Mac camera.")
    exit()

# Initialize and pair with the drone AFTER model loads
drone = Drone()
drone.pair()
print("Drone paired!")

drone.hover()
time.sleep(1)  # Give it a second to stabilize

print("Taking off...")
drone.takeoff()
drone.hover()
time.sleep(2)

def move_forward():
    print("Moving forward!")
    drone.set_pitch(10)
    drone.move(1)
    drone.hover()

def move_backward():
    print("Moving backward!") 
    drone.set_pitch(-10)
    drone.move(1)
    drone.hover()

print("Press 'SPACE' to detect objects, 'q' to land the drone and exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing Mac camera!")
            break

        # Show continuously recording video
        cv2.imshow("Mac Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # Detect when spacebar is pressed
            print("Processing frame for object detection...")
            img = cv2.resize(frame, (512, 512))
            img = np.expand_dims(img, axis=0).astype(np.uint8)

            # Run object detection
            results = model.signatures["serving_default"](images=tf.constant(img))
            result = {key: value.numpy() for key, value in results.items()}

            detection_classes = np.atleast_1d(result['output_3'][0].astype(int))  # Force int type
            detection_scores = np.atleast_1d(result['output_2'][0])
            detection_boxes = np.atleast_2d(result['output_0'][0])

            detected_objects = set()
            height, width, _ = frame.shape

            print(f"Detected class IDs: {detection_classes}")  # Debugging: Show detected object IDs

            for i in range(len(detection_classes)):
                class_id = int(detection_classes[i])  # Ensure class_id is an integer
                confidence = detection_scores[i]
                box = detection_boxes[i]

                # Debugging raw detection output
                print(f"Raw Detection - ID: {class_id}, Confidence: {confidence:.2f}")

                if confidence > 0.5:
                    object_name = coco_labels.get(class_id, "Unknown")  # Avoid crashes for unknown IDs
                    print(f"Detected: {object_name} with confidence {confidence:.2f}")

                    if object_name != "Unknown":
                        detected_objects.add(object_name)

                        ymin, xmin, ymax, xmax = box
                        xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, f"{object_name} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Take action based on detected objects
            if "apple" in detected_objects:
                move_forward()
            elif "cell phone" in detected_objects:
                move_backward()

        elif key == ord("q"):  # Exit when 'q' is pressed
            print("Landing drone...")
            drone.land()
            time.sleep(2)  # Allow time for safe landing
            break

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting emergency stop...")
    try:
        drone.emergency_stop()
    except Exception as stop_error:
        print(f"Emergency stop failed: {stop_error}")

finally:
    time.sleep(2)  # Ensure communication stops before closing
    drone.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Program complete.")
