from flask import Flask, jsonify
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import math

app = Flask(__name__)

# URLs for ESP32-CAMs
object_esp32cam_url = "http://192.168.101.52/cam-hi.jpg"  # Replace with actual URL
pothole_esp32cam_url = "http://192.168.101.101/cam-hi.jpg"  # Replace with actual URL

# Roboflow API setup for pothole detection
rf = Roboflow(api_key="3pr3aIfRs4KhkiRPT6w6")
project = rf.workspace().project("pothole-detection-bqu6s")
pothole_model = project.version(9).model

object_model = YOLO("yolo-Weights/yolov8n.pt")

# YOLO model setup for object detection (example placeholder)
# Replace with actual object detection model loading logic
object_classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                     "baseball bat",
                     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                     "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                     "teddy bear", "hair drier", "toothbrush"
                     ]

pothole_classNames = ["Pothole", "Drain Hole", "circle-drain-clean-", "drain-clean-", "drain-not-clean-", "hole",
                      "manhole", "sewer cover"]

object_Dict = {name: i + 1 for i, name in enumerate(object_classNames)}

# Map class names to indices for potholes

pothole_Dict = {name: i + 81 for i, name in enumerate(pothole_classNames)}

# Global variables to store predictions
object_prediction = None
pothole_prediction = None
prediction_result = None


@app.route('/make_object_prediction', methods=['GET'])
def make_object_prediction():
    global object_prediction
    try:
        # Open the video stream from the object ESP32-CAM
        cap = cv2.VideoCapture(object_esp32cam_url)
        ret, frame = cap.read()
        cap.release()

        if frame is None:
            print('No image received')
            return jsonify({'error': 'No image received'}), 500
        # Predict using the object detection model
        # Replace with the actual prediction logic for your YOLO model
        results = object_model(frame, stream=True)

        unique_objects = {}
        for r in results:
            for box in r.boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence >= 0.5:
                    class_name = object_classNames[int(box.cls[0])]
                    if class_name not in unique_objects:
                        unique_objects[class_name] = {
                            'ind': object_Dict[class_name],
                            'confidence': confidence
                        }

        objects_detected = list(unique_objects.values())
        if objects_detected:
            object_prediction = objects_detected
            return jsonify({'message': 'Object prediction stored successfully'})
        else:
            return jsonify({'error': 'No objects detected'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_object_prediction', methods=['GET'])
def get_object_prediction():
    global object_prediction
    if object_prediction is not None:
        result = object_prediction
        object_prediction = None  # Clear after retrieval
        return jsonify(result)
    else:
        return jsonify({'message': 'No new object prediction available'}), 200


@app.route('/make_pothole_prediction', methods=['GET'])
def make_pothole_prediction():
    global pothole_prediction
    try:
        # Open the video stream from the pothole ESP32-CAM
        cap = cv2.VideoCapture(pothole_esp32cam_url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("Frame Captured")

            # Predict using Roboflow pothole model
            prediction_group = pothole_model.predict(frame, confidence=40, overlap=30)

            if prediction_group.predictions:
                print("Entered in prep_group")
                predictions = prediction_group.predictions

                # Process predictions
                potholes_detected = [
                    {
                        'name': prediction['class'],
                        'ind': pothole_Dict[prediction['class']],
                        'confidence': prediction['confidence']
                    }
                    for prediction in predictions if prediction['confidence'] >= 0.5]

                prediction_result = potholes_detected
                if prediction_result:
                    # Convert prediction result to text
                    # prediction_text = ', '.join([f"{obj['name']} detected" for obj in prediction_result])
                    # Return the prediction as JSON
                    pothole_prediction = prediction_result
                    return jsonify({'message': 'Prediction stored successfully'})
                    # return jsonify(prediction_result)
                else:
                    return jsonify({'error': 'No objects detected'})
            else:
                return jsonify({'error': 'No objects detected 2'})
        else:
            return jsonify({'error': 'Failed to capture frame from webcam'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_pothole_prediction', methods=['GET'])
def get_pothole_prediction():
    global pothole_prediction
    if pothole_prediction is not None:
        result = pothole_prediction
        pothole_prediction = None  # Clear after retrieval
        return jsonify(result)
    else:
        return jsonify({'message': 'No new pothole prediction available'}), 200

#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)