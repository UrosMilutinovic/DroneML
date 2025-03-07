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

# Object labels (COCO dataset)
labels = {
    52: "apple",
    67: "cell phone"
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

# Ensure the drone is not already flying
drone.hover()
time.sleep(1)  # Give it a second to stabilize

# Take off safely
print("Taking off...")
drone.takeoff()
drone.hover()
time.sleep(2)  # Wait for 2 seconds before giving new commands

def move_forward():
    print("Moving forward!")
    drone.move(1, 0, 0, 0.5)  # Move forward at 50% speed for 0.5s
    drone.hover()  # Ensure it stops

def move_backward():
    print("Moving backward!")
    drone.move(-1, 0, 0, 0.5)  # Move backward at 50% speed for 0.5s
    drone.hover()  # Ensure it stops

print("Press 'q' to land the drone and exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing Mac camera!")
            break

        # Resize frame for model input and convert to uint8
        img = cv2.resize(frame, (512, 512))
        img = np.expand_dims(img, axis=0).astype(np.uint8)  # Convert to uint8

        # Run object detection
        detections = model.signatures["serving_default"](images=tf.constant(img))

        # Extract detection results
        detection_classes = np.atleast_1d(detections["output_3"].numpy()[0].astype(int))  # Ensure it's always an array
        detection_scores = np.atleast_1d(detections["output_2"].numpy()[0])  # Ensure it's always an array
        detection_boxes = np.atleast_2d(detections["output_0"].numpy()[0])  # Bounding boxes

        detected_objects = set()

        for i in range(len(detection_classes)):
            class_id = detection_classes[i]
            confidence = detection_scores[i]
            box = detection_boxes[i]  # Bounding box coordinates

            if confidence > 0.5:
                if class_id in labels:
                    object_name = labels[class_id]
                    detected_objects.add(object_name)
                    print(f"Detected: {object_name} with confidence {confidence:.2f}")
                else:
                    print(f"Unknown object detected with class ID {class_id} and confidence {confidence:.2f}")

                # Draw bounding box
                height, width, _ = frame.shape
                ymin, xmin, ymax, xmax = box
                xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = labels.get(class_id, f"ID {class_id}")
                cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Drone Movement Logic
        if "apple" in detected_objects:
            move_forward()
        elif "cell phone" in detected_objects:
            move_backward()

        # Display camera feed with bounding boxes
        cv2.imshow("Mac Camera", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Landing drone...")
            drone.land()  # Ensure landing
            break

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting emergency stop...")
    drone.emergency_stop()

finally:
    # Cleanup
    drone.close()  # Properly close the drone connection
    cap.release()
    cv2.destroyAllWindows()
    print("Program complete.")



'''import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from cordon_edu_sdk import CordonEDU  # Import Cordon EDU SDK (replace with actual import)

# Load object detection model
def load_model(model_url):
    print('Loading model...')
    model = hub.load(model_url)
    print('Model loaded!')
    return model

# Initialize Cordon EDU drone
def initialize_drone():
    drone = CordonEDU()
    drone.connect()
    return drone

# Control drone based on object position
def control_drone(drone, frame, detection_boxes):
    if detection_boxes.shape[0] > 0:  # If at least one object is detected
        box = detection_boxes[0]  # Get the first detected object
        ymin, xmin, ymax, xmax = box

        # Calculate the center of the bounding box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Control the drone based on object position
        if center_x < frame_width / 3:
            print("Move left")
            drone.rotate_left(30)  # Adjust according to Cordon EDU API
        elif center_x > 2 * frame_width / 3:
            print("Move right")
            drone.rotate_right(30)
        if center_y < frame_height / 3:
            print("Move up")
            drone.move_up(20)
        elif center_y > 2 * frame_height / 3:
            print("Move down")
            drone.move_down(20)

# Process frame for object detection
def process_frame(frame, model):
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image_np, axis=0)

    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}

    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    # Visualize detection results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        result['detection_boxes'][0],
        result['detection_classes'][0].astype(int),
        result['detection_scores'][0],
        category_index,  # Ensure category_index is defined
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
    )

    return frame, result['detection_boxes'][0]

# Main function
def main():
    model_url = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'
    model = load_model(model_url)

    # Initialize webcam and drone
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Cordon EDU drone
    drone = initialize_drone()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Process frame and run object detection
        processed_frame, detection_boxes = process_frame(frame, model)

        # Control drone based on detected object position
        control_drone(drone, processed_frame, detection_boxes)

        # Display processed frame
        cv2.imshow('Object Detection - Drone Control', processed_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Disconnect the drone
    drone.land()
    drone.disconnect()

if __name__ == '__main__':
    main()

#this is the chattted code that it gave me | more 'organied' format

import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from djitellopy import Tello  # Example for Tello drone (replace with your drone SDK)

# Load object detection model
def load_model(model_url):
    print('Loading model...')
    model = hub.load(model_url)
    print('Model loaded!')
    return model

# Load image into numpy array (not needed for webcam, but kept for reference)
def load_image_into_numpy_array(path):
    image = None
    if(path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

# Initialize Tello drone
def initialize_drone():
    drone = Tello()
    drone.connect()
    return drone

# Control drone based on object position
def control_drone(drone, frame, detection_boxes):
    if detection_boxes.shape[0] > 0:  # If there is at least one detected object
        box = detection_boxes[0]  # Get the first detected object
        ymin, xmin, ymax, xmax = box

        # Calculate the center of the bounding box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Control the drone based on the object position in the frame
        if center_x < frame_width / 3:
            print("Move left")
            drone.rotate_counter_clockwise(30)
        elif center_x > 2 * frame_width / 3:
            print("Move right")
            drone.rotate_clockwise(30)
        if center_y < frame_height / 3:
            print("Move up")
            drone.move_up(20)
        elif center_y > 2 * frame_height / 3:
            print("Move down")
            drone.move_down(20)

# Run the object detection and display results
def process_frame(frame, model):
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image_np, axis=0)

    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}

    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    # Visualize the detection results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        result['detection_boxes'][0],
        result['detection_classes'][0].astype(int),
        result['detection_scores'][0],
        category_index,  # You will need to define category_index for the model labels
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
    )

    return frame, result['detection_boxes'][0]

# Main function
def main():
    # Main function
    # Set model URL for object detection
    model_url = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'  # Modify with desired model
    model = load_model(model_url)


    # Initialize webcam and drone
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize the drone
    drone = initialize_drone()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Process the frame and run object detection
        processed_frame, detection_boxes = process_frame(frame, model)

        # Control the drone based on detected object position
        control_drone(drone, processed_frame, detection_boxes)

        # Display the processed frame
        cv2.imshow('Object Detection - Drone Control', processed_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Disconnect the drone
    drone.land()
    drone.end()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

-----------------------------------------------------------------------


import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from six.moves.urllib.request import urlopen
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


# @title Run this!!

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)


ALL_MODELS = {
'CenterNet HourGlass104 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',
'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
'CenterNet HourGlass104 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
'CenterNet HourGlass104 Keypoints 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
'CenterNet Resnet50 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
'CenterNet Resnet50 V1 FPN Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
'CenterNet Resnet101 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',
'CenterNet Resnet50 V2 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
'CenterNet Resnet50 V2 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
'EfficientDet D0 512x512' : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
'EfficientDet D1 640x640' : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
'EfficientDet D2 768x768' : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
'EfficientDet D3 896x896' : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1',
'SSD MobileNet v2 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
'SSD MobileNet V1 FPN 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
'SSD MobileNet V2 FPNLite 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
'SSD MobileNet V2 FPNLite 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',
'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',
'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',
'Faster R-CNN ResNet50 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
'Faster R-CNN ResNet50 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
'Faster R-CNN ResNet50 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',
'Faster R-CNN ResNet101 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
'Faster R-CNN ResNet101 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',
'Faster R-CNN ResNet101 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',
'Faster R-CNN ResNet152 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',
'Faster R-CNN ResNet152 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',
'Faster R-CNN ResNet152 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',
'Faster R-CNN Inception ResNet V2 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',
'Faster R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',
'Mask R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
}

IMAGES_FOR_TEST = {
  'Beach' : 'models/research/object_detection/test_images/image2.jpg',
  'Dogs' : 'models/research/object_detection/test_images/image1.jpg',
  # By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
  'Naxos Taverna' : 'https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
  'Beatles' : 'https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg',
  # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
  'Phones' : 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
  'Birds' : 'https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg',
}

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]

tf.get_logger().setLevel('ERROR')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

model_display_name = 'CenterNet HourGlass104 Keypoints 512x512' # @param ['CenterNet HourGlass104 512x512','CenterNet HourGlass104 Keypoints 512x512','CenterNet HourGlass104 1024x1024','CenterNet HourGlass104 Keypoints 1024x1024','CenterNet Resnet50 V1 FPN 512x512','CenterNet Resnet50 V1 FPN Keypoints 512x512','CenterNet Resnet101 V1 FPN 512x512','CenterNet Resnet50 V2 512x512','CenterNet Resnet50 V2 Keypoints 512x512','EfficientDet D0 512x512','EfficientDet D1 640x640','EfficientDet D2 768x768','EfficientDet D3 896x896','EfficientDet D4 1024x1024','EfficientDet D5 1280x1280','EfficientDet D6 1280x1280','EfficientDet D7 1536x1536','SSD MobileNet v2 320x320','SSD MobileNet V1 FPN 640x640','SSD MobileNet V2 FPNLite 320x320','SSD MobileNet V2 FPNLite 640x640','SSD ResNet50 V1 FPN 640x640 (RetinaNet50)','SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)','SSD ResNet101 V1 FPN 640x640 (RetinaNet101)','SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)','SSD ResNet152 V1 FPN 640x640 (RetinaNet152)','SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)','Faster R-CNN ResNet50 V1 640x640','Faster R-CNN ResNet50 V1 1024x1024','Faster R-CNN ResNet50 V1 800x1333','Faster R-CNN ResNet101 V1 640x640','Faster R-CNN ResNet101 V1 1024x1024','Faster R-CNN ResNet101 V1 800x1333','Faster R-CNN ResNet152 V1 640x640','Faster R-CNN ResNet152 V1 1024x1024','Faster R-CNN ResNet152 V1 800x1333','Faster R-CNN Inception ResNet V2 640x640','Faster R-CNN Inception ResNet V2 1024x1024','Mask R-CNN Inception ResNet V2 1024x1024']
model_handle = ALL_MODELS[model_display_name]

print('Selected model:'+ model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))

print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')

selected_image = 'Beach' # @param ['Beach', 'Dogs', 'Naxos Taverna', 'Beatles', 'Phones', 'Birds']
flip_image_horizontally = False
convert_image_to_grayscale = False

image_path = IMAGES_FOR_TEST[selected_image]
image_np = load_image_into_numpy_array(image_path)

# Flip horizontally
if(flip_image_horizontally):
  image_np[0] = np.fliplr(image_np[0]).copy()

# Convert image to grayscale
if(convert_image_to_grayscale):
  image_np[0] = np.tile(
    np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

plt.figure(figsize=(24,32))
plt.imshow(image_np[0])
plt.show()

# running inference
results = hub_model(image_np)

# different object detection models have additional results
# all of them are explained in the documentation
result = {key:value.numpy() for key,value in results.items()}
print(result.keys())

label_id_offset = 0
image_np_with_detections = image_np.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in result:
  keypoints = result['detection_keypoints'][0]
  keypoint_scores = result['detection_keypoint_scores'][0]

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections[0],
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

plt.figure(figsize=(24,32))
plt.imshow(image_np_with_detections[0])
plt.show()



# OpenCV webcam capture setup
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the resolution of the camera (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Load the object detection model
model_handle = ALL_MODELS['CenterNet HourGlass104 Keypoints 512x512']  # Use your desired model here
hub_model = hub.load(model_handle)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add a batch dimension to the image (required for the model)
    image_np = np.expand_dims(image_np, axis=0)

    # Run inference on the frame
    results = hub_model(image_np)

    # Process results from the object detection model
    result = {key: value.numpy() for key, value in results.items()}

    # Visualize the detected boxes and labels on the image
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    image_with_detections = image_np.copy()

    # Visualize the boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,  # You will need to define category_index for the model labels
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
    )

    # Convert the image with detections back to BGR for OpenCV to display
    image_with_detections_bgr = cv2.cvtColor(image_with_detections[0], cv2.COLOR_RGB2BGR)

    # Display the processed frame with detections
    cv2.imshow('Object Detection', image_with_detections_bgr)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Assuming a drone control library like 'dronekit' or 'djitellopy'
from djitellopy import Tello  # Example for Tello drone (replace with your drone SDK)

# Initialize drone object
drone = Tello()
drone.connect()

# Assuming you have detection boxes (detection_boxes)
if result['detection_boxes'][0].shape[0] > 0:  # If there is at least one detected object
    box = result['detection_boxes'][0][0]  # Get the first detected object
    ymin, xmin, ymax, xmax = box

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

    # Control the drone based on the object position in the frame
    if center_x < frame_width / 3:
        print("Move left")
        drone.rotate_counter_clockwise(30)
    elif center_x > 2 * frame_width / 3:
        print("Move right")
        drone.rotate_clockwise(30)
    if center_y < frame_height / 3:
        print("Move up")
        drone.move_up(20)
    elif center_y > 2 * frame_height / 3:
        print("Move down")
        drone.move_down(20)

# Continue with your regular detection code

if cv2.waitKey(1) & 0xFF == ord('q'):
  break
'''

