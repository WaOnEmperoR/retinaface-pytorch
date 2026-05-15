import cv2
import numpy as np
import json

def draw_detections(original_image, detections, vis_threshold, draw_landmark):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    
    if draw_landmark:
        landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    if draw_landmark:
        for box, score, landmark in zip(boxes, scores, landmarks):
            # Draw bounding box
            cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

            # Draw confidence score
            text = f"{score:.2f}"
            cx, cy = box[0], box[1] + 12
            cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

            # Draw landmarks
            for point, color in zip(landmark, LANDMARK_COLORS):
                cv2.circle(original_image, point, 1, color, 4)
    else:
        for box, score in zip(boxes, scores):
            # Draw bounding box
            cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

            # Draw confidence score
            text = f"{score:.2f}"
            cx, cy = box[0], box[1] + 12
            cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)
            
def write_detections(im_name, original_image, detections, vis_threshold):
    
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape
    
    data_obj = {"image": im_name}
    result = []
    annotations = []
    
    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    
    for box, score in zip(boxes, scores):
        x = (box[0] / img_width) * 100
        y = (box[1] / img_height) * 100
        width = ((box[2] - box[0]) / img_width) * 100
        height = ((box[3] - box[1]) / img_height) * 100

        value_obj = {
            "x" : round(float(x), 6),
            "y" : round(float(y), 6),
            "width" : round(float(width), 6),
            "height" : round(float(height), 6),
            "rotation": 0,
            "rectanglelabels": ["Other"]
        }
        
        result_obj = {
            "original_width": img_width,
            "original_height": img_height,
            "image_rotation": 0,
            "value": value_obj,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual"
        }
        
        result.append(result_obj)
    
    annotations.append({"result" : result})
    
    res = [{
        "data": data_obj,
        "annotations": annotations
    }]
    
    return json.dumps(res, indent=2)