## Installation

### Installation Options:


S1: Download dlib and install it with Python bindings:

- [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```
pip3 install face_recognition
```

Alternatively, you can try this library with [Docker](https://www.docker.com/), see [this section](https://github.com/ageitgey/face_recognition/blob/master/README.md#deployment).

If you are having trouble with installation, you can also try out a [pre-configured VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).



### Complete pipeline for Face Detection, Face Recognition and Emotion Detection
Refer to the notebook /src/facial_detection_recog_emotion.ipynb

We have trained an emotion detection model and put its trained weights at /emotion_detector_models

### Train your Emotion Detection Model
To train your own emotion detection model, Refer to the notebook /src/EmotionDetector_v2.ipynb

We have used an open source data set- Face Emotion Recognition (FER) from Kaggle and built a CNN to detect emotions.[Emotion Detection Data Set](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column. The training set consists of 28,709 examples.

You can load the pre-trained model and run it on an image using these lines of code below:

model = load_model("./emotion_detector_models/model.hdf5")
predicted_class = np.argmax(model.predict(face_image)
