"""
CS50 Final Project: util_dataCollection.py
Yanzhao Yang

Collects data (300x300 image) used to train model.

Code is self-written but largely based on the sources below.
Commented in-depth for learning purposes.

Sources Used:
https://www.youtube.com/watch?v=NZde8Xt78Iw&t=0s
"""

import cv2
import module_handTracking as hm
import numpy as np
import math
import time

# Initialize cap size parameters
capWidth, capHeight = 1920, 1080

# Open default camera (0) for video capturing
cap = cv2.VideoCapture(0)
# Set capWidth and capHeight
cap.set(3, capWidth)
cap.set(4, capHeight)

# Call hand_detector() and assign to object detector
detector = hm.hand_detector(maxHands=1)

# Data save directory
dataDirectory = "data/palm"
# Initialize dataCounter
dataCounter = 0

# Create infinite loop until program stopped
while True:
    # Read each frame from cap and return True and frame if successful
    success, image = cap.read()

    # Flip image
    image = cv2.flip(image, 1)

    # Find hands in image
    hands, image = detector.find_hands(image, flipType=False)

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

        cv2.imshow("imageCrop", imageCrop)
        cv2.imshow("imageWhite", imageWhite)

    # Show image
    cv2.imshow("Image", image)

    # Wait 1 ms before showing next image
    key = cv2.waitKey(1)

    if key == ord("s"):
        dataCounter += 1
        print(dataCounter)
        cv2.imwrite(f"{dataDirectory}/fist_{time.time()}.jpg", imageWhite)