"""
CS50 Final Project: app.py
Yanzhao Yang

Allows user to control sound using four gestures:
- Fist - do not play/stop playing
- Palm - play
- Palm moving up - increase volume
- Palm moving down - decrease volume

Code is self-written with reference to the sources below.
Commented in-depth for learning purposes.

Sources Used:
https://www.youtube.com/watch?v=NZde8Xt78Iw&t=0s
"""

import cv2
import module_handTracking as hm
import numpy as np
import math
import module_classification as cm
import pygame as pg

# Initialize cap size parameters
capWidth, capHeight = 1920, 1080

# Open default camera (0) for video capturing
cap = cv2.VideoCapture(0)
# Set capWidth and capHeight
cap.set(3, capWidth)
cap.set(4, capHeight)

# Call hand_detector() and assign to object detector
detector = hm.hand_detector(maxHands=1)
# Call classifier and assign to object classifier
classifier = cm.Classifier("model/keras_model.h5", "model/labels.txt")
# Initialize classification labels list
labels = ["fist", "palm"]
# Initialize labels index
index = 0

# Initiate pygame mixer
pg.mixer.init()
# Load audio file
sound = pg.mixer.music
sound.load("audio/440Hz.wav")
# Initialize soundOn to 0 and soundStatus to "stop"
soundOn = False
soundStatus = "stop"
# Set initial volume to 0.5
volume = 0.5
sound.set_volume(volume)
# Initialize distanceRange based on observed range during hand movement
distanceRange = [70, 700]
# Initialize volumeRange
volumeRange = [0, 1]

# Create infinite loop until program stopped
while True:
    # Read each frame from cap and return True and frame if successful
    success, image = cap.read()

    # Flip image
    image = cv2.flip(image, 1)

    # Find hands in image and return hands, imageOutput
    hands, imageOutput = detector.find_hands(image, drawAll=False, drawLandmarks=True, flipType=False)

    # If hands found
    if hands:
        # Extract and store hands[0] in hand
        hand = hands[0]

        # Extract bBox dimensions from hand and store in cropX, cropY, cropW, and cropH
        x, y, w, h = hand["bBox"]
        # Initialize cropOffset to ensure whole hand captured
        cropOffset = 20
        # Define cropY1, cropY2, cropX1, cropX2
        cropY1 = y - cropOffset
        cropY2 = y + h + cropOffset
        cropX1 = x - cropOffset
        cropX2 = x + w + cropOffset
        # Initialize cropSize
        cropSize = 300

        # Create cropSize * cropSize white image
        imageWhite = np.ones((cropSize, cropSize, 3), np.uint8) * 255

        # Crop image
        imageCrop = image[cropY1:cropY2, cropX1:cropX2]

        # Insert imageCrop centred onto imageWhite
        if imageCrop is not None:
            # Calculate cropHeight, cropWidth
            cropHeight = cropY2 - cropY1
            cropWidth = cropX2 - cropX1
            # If cropHeight is greater
            if cropHeight > cropWidth:
                # Calculate cropScale to scale cropHeight to cropSize
                cropScale = cropSize / cropHeight

                # Scale cropHeight and cropWidth
                cropHeight = math.floor(cropHeight * cropScale)
                cropWidth = math.floor(cropWidth * cropScale)

                # Resize imageCrop
                imageCrop = cv2.resize(imageCrop, (cropWidth, cropHeight))

                # Calculate cropSpace needed to centre image by width
                cropSpace = math.floor((cropSize - cropWidth) / 2)

                # Insert imageCrop onto imageWhite
                imageWhite[0:cropHeight, cropSpace:cropSpace + cropWidth] = imageCrop

                # Classify image
                prediction, index = classifier.get_prediction(imageWhite)

            # If cropWidth is greater
            else:
                # Calculate cropScale to scale cropHeight to cropSize
                cropScale = cropSize / cropWidth

                # Scale cropHeight and cropWidth
                cropHeight = math.floor(cropHeight * cropScale)
                cropWidth = math.floor(cropWidth * cropScale)

                # Resize imageCrop
                imageCrop = cv2.resize(imageCrop, (cropWidth, cropHeight))

                # Calculate cropSpace needed to centre image by height
                cropSpace = math.floor((cropSize - cropHeight) / 2)

                # Insert imageCrop onto imageWhite
                imageWhite[cropSpace:cropSpace + cropHeight, 0:cropWidth] = imageCrop

                # Classify image
                prediction, index = classifier.get_prediction(imageWhite)

        # Find landmarks in image
        landmarks = detector.find_landmarks(image, draw=False)
        # Set p1 to landmark 0 and p2 to bottom of image directly below landmark 0
        p1 = landmarks[0][1], landmarks[0][2]
        p2 = landmarks[0][1], capHeight

        # Find perpendicular distance b/w landmark 0 and bottom of image
        distance, info = detector.find_distance(p1, p2)

        # If palm detected and not soundOn
        if labels[index] == "palm" and not soundOn:
            # Play sound
            sound.play()

            # Set soundOn to True
            soundOn = True

            # Set audioStatus to play
            soundStatus = "play"
        # If palm detected and soundOn
        elif labels[index] == "palm" and soundOn:
            # Calculate volume using linear interpolation based on distance
            volume = np.interp(distance, distanceRange, volumeRange)

            # Set new volume
            sound.set_volume(volume)
        # If fist detected and soundOn
        elif labels[index] == "fist" and soundOn:
            sound.stop()
            soundOn = False
            soundStatus = "stop"

        print(soundStatus)

        # Extract perpendicular line info from info
        x1, y1, x2, y2, cx, cy = info

        # Draw bounding box and put label
        cv2.rectangle(imageOutput, (cropX1, cropY1), (cropX2, cropY2), (0, 0, 255), 2)
        cv2.putText(imageOutput, labels[index], (cropX1, cropY1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Draw distance between landmark 0 and bottom of image
        cv2.circle(imageOutput, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
        cv2.line(imageOutput, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Show image
    cv2.imshow("imageOutput", imageOutput)

    # Wait 1 ms before showing next image
    cv2.waitKey(1)