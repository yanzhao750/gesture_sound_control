"""
CS50 Final Project: module_classification.py
Yanzhao Yang

Classifies hand symbol as palm or fist based on pre-trained Teachable Machine model.

Code is self-written but largely based on the sources below.
Commented in-depth for learning purposes.

Sources used:
https://github.com/cvzone/cvzone/blob/master/cvzone/ClassificationModule.py
"""

import tensorflow.keras
import numpy as np
import cv2

class Classifier:
    def __init__(self, pathModel, pathLabels=None):
        """
        :param pathModel: Path to model
        :param pathLabels: Path to labels; default None
        """
        # Store passed model path
        self.pathModel = pathModel

        # Suppress scientific notations for clarity
        np.set_printoptions(suppress=True)
        # Load model
        self.model = tensorflow.keras.models.load_model(self.pathModel)

        # Create array to feed keras model
        # Length or number of images that can be put into is determined by first position in tuple shape (1)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.pathLabels = pathLabels
        if self.pathLabels:
            labelFile = open(self.pathLabels, "r")
            self.labels = []
            for line in labelFile:
                strippedLine = line.strip()
                self.labels.append(strippedLine)
            labelFile.close()
        else:
            print("No labels found")

    def get_prediction(self, image, draw=True, pos=(50, 50), scale=2, color=(0, 0, 255)):
        """

        :param image: Image to make prediction on
        :param draw: Puts label if True; default True
        :param pos: Position of label
        :param scale: Scale of label
        :param color: Color of label

        :return: list(prediction[0]):
        :return: index"
        """
        # Resize image to 224x224
        image224 = cv2.resize(image, (224, 224))

        # Turn image into numpy array
        imageArray = np.array(image224)

        # Normalize image
        imageArrayNormalized = (imageArray.astype(np.float32) / 127.0) - 1

        # Load  image into array
        self.data[0] = imageArrayNormalized

        # Run inference
        prediction = self.model.predict(self.data, verbose=0)
        index = np.argmax(prediction)

        if draw and self.pathLabels:
            cv2.putText(image, str(self.labels[index]), pos,
                        cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), index

"""
The following section is for standalone module testing. It will not be used if the module is imported.
"""
def main():
    cap = cv2.VideoCapture(0)
    classifierMask = Classifier('model/keras_model.h5', 'model/labels.txt')
    while True:
        _, image = cap.read()
        prediction = classifierMask.get_prediction(image)
        # print(prediction)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()