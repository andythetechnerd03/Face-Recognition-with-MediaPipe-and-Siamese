import time
from typing import List, Tuple
import cv2
import mediapipe as mp
import torch
import numpy as np
from resources.model import to_numpy
import logging
from resources.configs import transformers

mp_face_detection = mp.solutions.face_detection

# Make a FaceDetector class
def extract_image_by_bounding_box(image: np.ndarray,
                                  x_min: int,
                                  y_min: int,
                                  width: int,
                                  height: int) -> torch.Tensor:
    """
    Extracts the image from the bounding box given the coordinates and the width and height.
    :param image: The image to extract from, in the form of numpy array
    :param x_min: The x coordinate of the top left corner of the bounding box
    :param y_min: The y coordinate of the top left corner of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :return: The extracted image in the form of tensor and in the shape of (channels, height, width)
    """
    return torch.Tensor(image[y_min:y_min + height, x_min:x_min + width]).permute(2, 0, 1)


def get_name_from_recognition(database: np.ndarray,
                 frame: np.ndarray,
                 labels: List[str]):
    """
    Recognize the identity of the face in the frame given the database of faces.
    Using Euclidean distance, the face in the frame is compared to the faces in the database, and the lowest is chosen.
    The formula is as follows:
    $`d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}`$
    :param database: a numpy array of the database of faces of size ```(number of faces, channels, height, width)```.
    :param frame: a numpy array of the face in the frame of size ```(channels, height, width)```
    :param labels: a list of the labels of the faces in the database
    :return: the name of the person and the Euclidean distance between the face and the face in the database
    """
    dist_matrix = torch.cdist(torch.Tensor(database), torch.Tensor(frame), p=2)
    argmin_matrix = torch.argmin(dist_matrix, dim=0)
    min_value = torch.min(dist_matrix, dim=0)[0].item()
    name = labels[argmin_matrix]
    return name, min_value


class FaceDetector:
    """
    A class to detect faces using MediaPipe Face Detection.
    Model Selection: 0 for short-range, works well with detecting close faces;
                     1 for long-range, works well with detecting far-away faces.
    Min Detection Confidence: The minimum confidence value ([0.0, 1.0]) from the face detection model for the detection,
                                the lower the value, the more false positives.
    Detector: The MediaPipe Face Detection object itself with all the above specifications.
    """

    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.detector = mp_face_detection.FaceDetection(
            model_selection=self.model_selection, min_detection_confidence=self.min_detection_confidence)

    def __call__(self, image):
        results = self.detector.process(image)
        return results

    def recognize(self, format, model, image, labels, database, threshold=0.8):
        """
        Recognize the face in the image given the model, the database, and the threshold.
        :param format: str: format of the model, either "torch" or "onnx", "tensorflow" don't be sad!
        :param model: the model to use for recognition, either PyTorch or ONNX, must be aligned with the format.
        :param image: the image to recognize the face from, in the form of numpy array.
        :param labels: a list of the labels of the faces in the database.
        :param database: the database of faces in the form of tensor.
        :param threshold: the threshold to determine whether the face is known or unknown, has to be positive.
        :return: the results of the face detection and the face in the image.
        """
        if format != "torch" and format != "onnx":
            raise ValueError("format must be either 'torch' or 'onnx', 'tensorflow' don't be sad!")
        if threshold < 0:
            raise ValueError("what do you mean by negative threshold?")

        results = self.detector.process(image)
        # Get the bounding box
        if results.detections:
            for detection in results.detections:
                xmin = max(0,int(detection.location_data.relative_bounding_box.xmin * image.shape[1]))
                ymin = max(0,int(detection.location_data.relative_bounding_box.ymin * image.shape[0]))
                width = max(0,int(detection.location_data.relative_bounding_box.width * image.shape[1]))
                height = max(0,int(detection.location_data.relative_bounding_box.height * image.shape[0]))

                # Extract the face from the image
                face = extract_image_by_bounding_box(image, xmin, ymin, width, height)
                # Apply the transformations, and fill the extra dimension
                face = transformers(face)
                if face.ndim == 3:
                    face = face.unsqueeze(0)

                # Recognize the face in two formats: PyTorch and ONNX
                if format == "torch":
                    with torch.inference_mode():
                        output_1, output_2 = model(database, face)
                elif format == "onnx":
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(database),
                              model.get_inputs()[1].name: to_numpy(face)}
                    output_1, output_2 = model.run(None, ort_inputs)

                id, dist = get_name_from_recognition(output_1, output_2, labels)

                # Draw the bounding box
                draw_boxes(image, xmin, ymin, width, height, id, dist, threshold)

            return results, face
        else:
            return results, None


def draw_boxes(image: np.ndarray,
               xmin: int,
               ymin: int,
               width: int,
               height: int,
               name: str,
               dist: float,
               threshold: float) -> None:
    """
    Draw the bounding box on the image, given the coordinate of the upper-left corner and the width and height,
    as well as the identifier and the Euclidean distance from the face in the frame to the faces in the database.
    :param image: a numpy array of the image of size (height, width, channels)
    :param xmin: The x coordinate of the top left corner of the bounding box
    :param ymin: The y coordinate of the top left corner of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :param name: The identifier of the person
    :param dist: The Euclidean distance between the face and the face in the database
    :param threshold: The threshold to determine whether the face is known or unknown
    :return: Nothing
    """
    start_point = (xmin, ymin)
    end_point = (xmin + width, ymin + height)
    # Determine the name of the person based on the distance and threshold.
    name = name if dist < threshold else "Unknown"
    # Determine the color of the box.
    color = (36, 255, 12) if dist < threshold else (255, 0, 0)
    # Annotation
    image = cv2.rectangle(image, start_point, end_point, color, 1)
    image = cv2.putText(image, f"{name} {'{:.2f}'.format(dist)}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, 2)
    return image

def count_fps(prev_frame_time: float,
              new_frame_time: float) -> tuple[int, float, float]:
    """
    Count the FPS of the video.
    :param prev_frame_time: the time of the previous frame
    :param new_frame_time: the time of the current frame
    :return: the FPS
    """
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    return int(fps), prev_frame_time, new_frame_time

# For webcam input:
def webcam_recognition(format: str,
                       face_detection: FaceDetector,
                       model,
                       labels: List[str],
                       database: torch.Tensor,
                       fps: bool,
                       source: int,
                       threshold=0.8):
    """
    Perform face recognition on the webcam, using cv2 VideoCapture functionality.
    :param source: int, the source of the video, 0 for webcam, or the path to the video file.
    :param fps:  bool, whether to show the FPS or not
    :param face_detection: object, the MediaPipe Face Detection to use for face detection.
    :param format: str, format of the model, either "torch" or "onnx", "tensorflow" fans don't be sad!
    :param model: object, the model to use for recognition, either PyTorch or ONNX, must be aligned with the format.
    :param labels: Tuple or List, a list of the labels of the faces in the database.
    :param database: torch.Tensor, the database of faces in the form of tensor.
    :param threshold: float, the threshold to determine whether the face is known or unknown, has to be positive.
    :return: None
    """
    cap = cv2.VideoCapture(source)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    with face_detection.detector as detector:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logging.warning("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results, face = face_detection.recognize(format, model, image, labels, database, threshold)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Calculating the fps

            fps, prev_frame_time, new_frame_time = count_fps(prev_frame_time, new_frame_time)
            cv2.putText(image, f"FPS: {fps}", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow('Face Recognition', image)

            # Quit the program when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


