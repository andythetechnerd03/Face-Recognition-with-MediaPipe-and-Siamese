from resources.face_detection import FaceDetector, webcam_recognition
from resources.model import load_model, SiameseNetwork
from resources.configs import transformers
import argparse
from resources.face_dataset import initialize_database
import logging

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, help="The folder of all the faces in the database",
                      default="images")
    args.add_argument("--model", type=str, default="resources/siamese_16.onnx")
    args.add_argument("--threshold", type=float, default=0.8)
    args.add_argument("--min_detection_confidence", type=float, default=0.5)
    args.add_argument("--model_selection", type=int, default=0)
    args.add_argument("--fps", type=bool, help="Whether to turn on FPS tracking", default=False)
    args = args.parse_args()

    if args.model.endswith(".onnx"):
        format = "onnx"
    elif args.model.endswith(".pth"):
        format = "torch"
    else:
        raise ValueError("The model must be either .onnx or .pth")

    # Load the model
    model = load_model(format, args.model)
    logging.info('Model loaded successfully')

    # Initialize the database
    database, labels = initialize_database(args.path, transformers)
    logging.info("Database initialized successfully")

    # Initialize the face detector
    face_detector = FaceDetector(args.model_selection, args.min_detection_confidence)
    logging.info("Face detector initialized successfully")

    # Perform face recognition
    webcam_recognition(format, face_detector, model, labels, database, args.fps, args.threshold)
    logging.info("Face recognition completed successfully")
