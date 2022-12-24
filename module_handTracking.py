"""
CS50 Final Project: module_handTracking.py
Yanzhao Yang

Uses MediaPipe Hands solution to find hands, landmark locations, and distance between two points.

Code is self-written but largely based on the sources below.
Commented in-depth for learning purposes.

Sources Used:
https://www.youtube.com/watch?v=NZde8Xt78Iw&t=0s
https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py
https://google.github.io/mediapipe/solutions/hands.html
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np

class hand_detector():
    # Define special method __init__ and initialize default parameters
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: Treats input as video stream if False; default False
        :param maxHands: Maximum number of hands to detect; default 2
        :param modelComplexity: Landmark accuracy and inference latency increase with complexity; 0 or 1
        :param detectionCon: Minimum confidence from hand detection model to be considered successful; [0.0, 1.0]
        :param minTrackCon: Minimum confidence from landmark tracking model to be considered successful, 
                            otherwise hand detection will be invoked on next input image; default 0.5
        """

        # Parameters
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        # Create object self.mpHands by calling module mp.solutions.hands
        self.mp_hands = mp.solutions.hands
        # Create object self.hands using class self.mpHands.Hands and pass parameters
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.minTrackCon)
        # Create object self.mpDraw by calling module mp.solutions.drawing_utils
        self.mpDraw = mp.solutions.drawing_utils

    # Define user method find_hands
    def find_hands(self, image, drawAll=True, drawLandmarks=False, flipType=True):
        """
        Finds hands in an RGB image.

        :param image: Image to find hands in; passed as BGR format from opencv, converted to RGB for mediapipe
        :param draw: Draws hands if True; default True

        :return hands: List of dicts with all hands
        :return image: image with all drawings if drawAl=True, or landmark drawings only if drawLandmarks=True
        """

        # Convert image to RGB colour space
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        self.results = self.hands.process(imageRGB)

        # Initialize empty list hands
        hands = []

        # Extract and store shape of image
        height, width, channel = image.shape

        # If hand landmarks detected
        if self.results.multi_hand_landmarks:
            # For each handType and handLandmarks
            for handType, handLandmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                # Initialize empty dict hand
                hand = {}

                # Initialize empty lists landmarks, xList, and yList
                landmarks = []
                xList = []
                yList = []

                # For each landmark
                for id, landmark in enumerate(handLandmarks.landmark):
                    # Calculate the pixel coordinates
                    px, py, pz = int(landmark.x * width), int(landmark.y * height), int(landmark.z * channel)

                    # Append px, py, and pz to lists
                    landmarks.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # Draw a bounding box around hand
                # Find min, max values for x, y
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)

                # Calculate box dimensions
                boxW, boxH = xMax - xMin, yMax - yMin
                # Create bBox list
                bBox = xMin, yMin, boxW, boxH

                # Calculate centrepoint of box
                cx, cy = bBox[0] + (bBox[2] // 2), bBox[1] + (bBox[3] // 2)

                # Insert lists into dict hand
                hand["landmarks"] = landmarks
                hand["bBox"] = bBox
                hand["centre"] = (cx, cy)

                # Switch hand classification if image flipped
                if flipType:
                    if handType.classification[0].label == "Right":
                        hand["type"] = "Left"
                    else:
                        hand["type"] = "Right"
                else:
                    hand["type"] = handType.classification[0].label

                # Append hand to hands
                hands.append(hand)

            # Draw landmarks, Box and put label text if drawAll=True
            if drawAll:
                # Draw landmarks
                self.mpDraw.draw_landmarks(image, handLandmarks, self.mp_hands.HAND_CONNECTIONS)
                # Draw bBox
                cv2.rectangle(image, (bBox[0] - 20, bBox[1] - 20),
                              (bBox[0] + bBox[2] + 20, bBox[1] + bBox[3] + 20),
                              (0, 0, 255), 2)
                # Put label text
                cv2.putText(image, hand["type"], (bBox[0] - 30, bBox[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            # Draw landmarks only if drawLandmarks=True
            elif drawLandmarks:
                # Draw landmarks
                self.mpDraw.draw_landmarks(image, handLandmarks, self.mp_hands.HAND_CONNECTIONS)

        # If draw=True, return list hands and image
        if drawAll or drawLandmarks:
            return hands, image
        # If draw=False, return list hands
        else:
            return hands

    # Define user method find_landmarks to find the position of each landmark
    def find_landmarks(self, image, handIndex=0, draw=True):
        """
        Finds the position of each landmark.

        :param image: Image to find landmarks and draw on
        :param handIndex: Index of hand to find landmarks on
        :param draw: Draws landmarks if True; default True

        :return landmarks: List of lists with each landmark's pixel coordinates
        :return image: image with drawing if draw=True
        """
        # Declare empty list landmarks
        landmarks = []

        # If hand landmarks detected
        if self.results.multi_hand_landmarks:
            # Store hand landmarks in hand
            hand = self.results.multi_hand_landmarks[handIndex]

            # For each landmark in hand.landmark, enumerated
            for landmarkId, landmark in enumerate(hand.landmark):  # Enumerate returns index and value at index
                # Store the height, width (dimensions) and channel (colourspace) of the image
                height, width, channel = image.shape

                # Calculate the pixel coordinates
                px, py = int(landmark.x * width), int(landmark.y * height)

                # Append landmarkId, cx, cy
                landmarks.append([landmarkId, px, py])

                # Draw landmarks if draw=True
                if draw:
                    cv2.circle(image, (px, py), 5, (0, 0, 255), cv2.FILLED)

        # If draw=True, return list landmarks and image
        if draw:
            return landmarks, image
        # If draw=False, return list landmarks
        else:
            return landmarks

    # Define user method find_distance
    def find_distance(self, p1, p2, image=None):
        """
        Finds distance between two landmarks based on landmark index numbers.

        :param p1: Location tuple 1
        :param p2: Location tuple 2
        :param image: Image to find landmark indexes and to draw on

        :return distance: Distance between p1 and p2
        :return info: Tuple with pixel coordinates of p1, p2, and centrepoint
        :return image: Image with drawing if draw=True
        """

        # Initialize x, y using p
        x1, y1 = p1
        x2, y2 = p2

        # Find centre between p1, p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate distance
        distance = math.hypot(x2 - x1, y2 - y1)

        # Store calculated values in tuple info
        info = (x1, y1, x2, y2, cx, cy)

        # If image passed draw circles on p1, p2, c and line b/w p1, p2
        if image is not None:
            cv2.circle(image, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)

            return distance, info, image
        # If image not passed
        else:
            return distance, info

"""
The following section is for standalone module testing. It will not be used if the module is imported.
"""
def main():

    # Initialize timePrevious and timeCurrent to 0 for fps
    timePrevious = 0
    timeCurrent = 0

    # Initialize cap size parameters
    capWidth, capHeight = 1920, 1080

    # Open default camera (0) for video capturing
    cap = cv2.VideoCapture(0)
    # Set capWidth and capHeight
    cap.set(3, capWidth)
    cap.set(4, capHeight)

    # Call hand_detector() and assign to object detector
    detector = hand_detector()

    # Create infinite loop until program stopped
    while True:
        # Read each frame from cap and return True and frame if successful
        success, image = cap.read()

        # Find hands in image frame
        hands, image = detector.find_hands(image)

        # Find landmarks in image frame
        landmarks = detector.find_landmarks(image, draw=False)

        # Find perpendicular distance between landmark 0 and bottom of image
        if hands:
            # Extract x1, y1, x2 from landmark[0] and set y2 to capHeight
            x1, y1 = landmarks[0][1], landmarks[0][2]
            x2, y2 = landmarks[0][1], capHeight

            # Set p1 to landmark 0 and p2 to bottom of image directly below landmark 0
            p1 = x1, y1
            p2 = x2, y2

            # Find distance, info and image using find_distance
            distance, info, image = detector.find_distance(p1, p2, image)

            # Calculate distanceNorm using linear interpolation
            distanceNorm = np.interp(distance, [100, 700], [0, 1])
            print(distanceNorm)

        # Initialize timeCurrent
        timeCurrent = time.time()
        # Calculate fps
        fps = 1 / (timeCurrent - timePrevious)
        # Update timePrevious to timeCurrent
        timePrevious = timeCurrent

        # Draw fps value onto image
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # Show image
        cv2.imshow("Image", image)
        # Wait 1 ms before showing next image
        cv2.waitKey(1)


if __name__ == "__main__":
    main()