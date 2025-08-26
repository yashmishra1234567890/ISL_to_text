
# Sign Language Detection using Webcam and Deep Learning

A simple Python script to detect sign language gestures in real-time from your webcam and display the recognized sign as text.


## How to Use
1. Make sure you have Python 3.7 or newer.
2. Install these packages:
  ```bash
  pip install opencv-python mediapipe tensorflow numpy
  ```
3. Put `model.h5` and `labels.txt` in the same folder as `run.py`.
4. Run:
  ```bash
  python run.py
  ```
5. Show a sign to your webcam. The app will display the detected word on the screen.
6. Press `q` to quit.

## Supported Signs
- Absent
- Bacteria
- Beautiful
- Cabbage
- Call
- Enemy
- Face
- Fall
- Happy
- Luck

## Files
- `simple_run.py`: Runs the app
- `labels.txt`: List of words
- `model.h5`: The trained model
- `training.ipynb`: For training your own model
- `X_data.npy`, `y_labels.npy`: Training data

## Train Your Own Model
Use `training.ipynb` to create and train a new model if you want to add more signs.

## Libraries Used
- Mediapipe
- TensorFlow
- OpenCV
4. Press `q` to quit.


