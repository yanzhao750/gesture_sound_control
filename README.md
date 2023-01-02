#cs50 #python 
# Sound Control Using Hand Gestures

### Video Demo
https://youtu.be/BlZeWKR83qk

### Project Summary
This project allows the user to control sound using simplified gestures inspired by those of a music conductor using computer vision. The example sound used is [A440](https://en.wikipedia.org/wiki/A440_(pitch_standard)) and four gestures are supported:
* Fist - do not play/stop playing
* Palm - play
* Palm moving up - increase volume
* Palm moving down - decrease volume
When app.py is run, no sound initially plays. If a palm is detected, the sound plays, and if the hand is moved up or down while a palm is continuously detected, the volume increases or decreases accordingly. If a fist is detected, the sound stops playing.

[OpenCV](https://OpenCV.org/) was used for video capture and image extraction, manipulation, and annotation; [MediaPipe Hands](https://google.github.io/MediaPipe/solutions/hands.html) was used to detect hands and hand landmarks; [Teachable Machine](https://teachablemachine.withgoogle.com/) was used to generate and train the custom fist and palm detection model; [Pygame](https://www.pygame.org/news) was used to control sound playback and volume. Each program or module and the methods contained therein is described below.

### Module Descriptions
#### app.py
This is the main application. It opens a 1920x1080 video feed using OpenCV, and initializes the hand `detector`, hand symbol `classifier`, hand symbol `labels`, and sound functions from Pygame. An infinite loop is created, wherein if a hand is detected, it is cropped and fitted to a 300x300 image with a white background; this standardizes the image, which improves model performance, before it is passed to `classifier.get_prediction`. The returned values `prediction` and `index` describe the prediction results and the hand symbol type respectively. After finding landmarks using `detector.find_landmarks`, the coordinates of `landmark[0]` (the hand-wrist interface) and the bottom of the image directly below `landmark[0]` are found, and passed to `detector.find_distance`. The returned `distance` and `info` are used to determine the perpendicular distance from `landmark[0]` to the bottom of the image, which represents the hand height, and is used to interpolate the volume. Based on the hand symbol type and hand height, the sound is played or stopped, and the volume is increased or decreased. The landmarks, bounding box (based on the cropped image), hand symbol type, and distance line are then drawn, and the image is shown. 

#### module_classification.py
Uses the Teachable Machine generated model to classify detected hands as either fist or palm.

##### Method `get_prediction`
Parameters:
* image: Image to make prediction on
* draw: Puts label if True; default True
* pos: Position of label
* scale: Scale of label
* color: Color of label
Returns:
* list(prediction[0]): list of information for prediction[0]
* index: index of hand symbol type

Loads the Teachable Machine generated model into TensorFlow. Resizes the image to 224x224 and creates a normalized `np.array` from the image. Passes the array via `self.data` (an `np.ndarray`) into the TensorFlow model, and stores the returned `prediction` and `index` (from `argmax(prediction)`). Draws a label if `draw=True`.

#### module_handTracking.py
Uses MediaPipe to find hands, landmark locations, and the distance between two points.

##### Method `find_hands(self, image, drawAll=True, drawLandmarks=False, flipType=True)`
Parameters:
* image: Image to find hands in; passed as BGR format from OpenCV, converted to RGB for MediaPipe
* drawAll: Draws landmarks, bounding box, and hand type if True; default True
* drawLandmarks: Draws landmarks only if True; default False
* flipType: Inverses hand type if True (assumes image is flipped horizontally); default False
Returns:
* hands: List of dicts with all hands
* image: Image with all drawings if drawAll=True, or landmark drawings only if drawLandmarks = True

Converts the passed image from BGR to RGB. Processes the image using `self.mp_hands.Hands.process` and stores the result in `self.results`. If hands are detected (`self.results.multi_hand_landmarks not False`), iterates with `zip()` through each `handType` (left or right) and `handLandmarks`, and calculates the pixel coordinates of each landmark as well as the bounding box location, dimensions, and centre point. Checks if `flipType=True` and inverses the hand type if needed, and inserts landmark, bounding box, and hand type information into `hand`, and appends to `hands`. Draws landmarks, bounding box, and hand type if `drawAll=True`, or landmarks only if `drawLandmarks=True`.

##### Method `find_landmarks(self, image, handIndex=0, draw=True)`
Parameters:
* image: Image to find landmarks on
* handIndex: Index of hand to find landmarks on
* draw: Draws landmarks if True; default True
Returns: 
* landmarks: List of lists with each landmark's pixel coordinates
* image: Image with drawing if draw=True

If hands are detected, extracts results from `self.results.multi_hand_landmarks` and stores in `hand`. Iterates with `enumerate()` through each `hand.landmark`, calculates the pixel coordinates of each landmark, and appends the `landmarkId` (returned by `enumerate()`) along with the pixel coordinates to list `landmarks[]`. Draws each landmark if `draw=True`.

##### Method `find_distance(self, p1, p2, image=None)`
Parameters:
* p1: location tuple 1
* p2: location tuple 2
Returns:
* distance: Distance between p1 and p2
* info: Tuple with pixel coordinates of p1, p2, and the centre point
* image: Image with drawing if image passed

Extracts the pixel coordinates of `p1` and `p2` and calculates the centre point coordinates. Finds `distance` between `p1` and `p2` using `math.hypot()`. If an image is passed (`image is not None`), draws `p1`, `p2`, and the centre point, and a line intersecting all three points.

#### util_dataCollection.py
This utility is used purely for data collection to train the model, and is not used when app.py is run. It shares a similar structure to app.py in that it opens a 1920x1080 video feed using OpenCV and initializes the hand `detector`, and creates an infinite loop wherein if a hand is detected, it is cropped and fitted to a 300x300 image with a white background. To save the image, `cv2.waitKey()` is set to `s` which, if pressed, will write the cropped image to a data folder on disk.

### Usage
Run app.py. Wait for the video capture feed to load in a separate window (may take up to 1 min). Use the gestures described in Project Summary to control the sound. Stop app.py once complete.

### Acknowledgement:
While all code is self-written, external sources were used extensively to understand the topics required. Most notably [Murtaza's Workshop](https://www.youtube.com/@murtazasworkshop) on Youtube and [CVZone](https://github.com/cvzone/cvzone) on Github were referenced, along with numerous forums and the respective documentation sites of OpenCV, MediaPipe, and Pygame. Specific videos and repositories referenced to write each Python file are linked in each file's header comments under Sources Used.
