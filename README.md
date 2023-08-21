# Face Recognition with MediaPipe and Siamese
This is a project on an application on Face Recognition using some the popular frameworks out there, from the MediaPipe from Google for [Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detector), and [Siamese Network](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi5odWl9u2AAxVzmlYBHUPVDzwQFnoECA0QAQ&url=https%3A%2F%2Fwww.cs.cmu.edu%2F~rsalakhu%2Fpapers%2Foneshot1.pdf&usg=AOvVaw0gKET0McCdIoco9UX2KcsE&opi=89978449).
## Specification
- Face Detection: MediaPipe Face Detection Module based on BlazeFace, which can achieve faster inference than traditional Haar-Cascade detector, but still not as strong as some more sophiscated model such as YOLO.
- Face Recognition: Siamese Network trained on AT&T Face Dataset. It will probably work better if you can find a bigger dataset, but this should work mostly OK.
- CPU: Intel Core i5 Gen 11
- Frameworks used: **PyTorch** for the base model, **OpenCV** for camera usage, **ONNX** for an even faster inference.
## Screenshot
![Screenshot (3173)](https://github.com/andythetechnerd03/Face-Recognition-with-MediaPipe-and-Siamese/assets/101492362/8fa6c291-1c75-43d0-b165-34d1cb98180e)

## How to Run
1. Install the dependencies
   ```pip install -r requirements.txt```
2. Take a picture of your face and put it in the `images` folder, make sure to rename it to `{your_name}.jpg`. It can be any resolution, but it must be square, otherwise the model might have a hard time learning it, since it will be scaled down to `(256x256)`. Tests have shown that
high-quality pictures help the model tremendously.
3. Run the recognizer
```python main.py --path --model --threshold --min_detection_confidence --model_selection --source --fps``` </br>
`--path`: folder to the images, default to `images/` </br>
`--model`: path to the model, right now the model supports `torch` and `onnx` models only, you can adjust the code to accept `tensorflow` models as well. </br>
`--threshold`: the minimum distance to recognize the image as "Unknown", below that it will output the name, above that it will be "Unknown", here we use Euclidean distance. </br>
`--min_detection_confidence`: Confidence in detecting a face, in the range `[0,1]`, the higher it is the less likely it is to detect a face. </br>
`--model_selection`: an int of either `0` or `1`: `0` mean using a short-range model (perfect for up-close shot), `1` for long-range model (perfect for detecting far-away faces) </br>
`--source`: the source for the camera: `0` is the default for Windows webcam, or any url if you use an IP Camera. </br>
`--fps`: bool, whether to display FPS Count. </br>
## Torch vs ONNX
![pytorchvsonnx](https://github.com/andythetechnerd03/Face-Recognition-with-MediaPipe-and-Siamese/assets/101492362/090fb5d7-ee57-466c-8238-425bb189da48)
ONNX typically performs smoother than PyTorch because of its optimization. ONNX runs around 15-25 FPS, while PyTorch runs around 10 FPS. This is, however, just for CPU inference; GPU inference might be close on both.
## Things to do
- Add an option for GPU inference.
- Add an option for Tensorflow.
- Upgrade the Siamese model with some better training data.
- And more which I will list out when I remember...
## Note
- Unlike most repositories, I focus a lot on code quality and readability, so I included many docstrings and loggings for the nerds out there. It is just mindful thinking when others read your code and understand what is going on.
## Credits: An Dinh (Andy)

