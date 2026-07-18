OpenCV Video Background Removal Tutorial
Theory
Background subtraction is a fundamental technique in computer vision used to separate foreground objects from the background in video sequences. This is particularly useful for applications like:

Motion detection and tracking

Object detection in surveillance systems

Gesture recognition

Video conferencing (virtual backgrounds)

Traffic monitoring

How Background Subtraction Works
Background Modeling: The algorithm learns what the static background looks like over time

Foreground Detection: Pixels that differ significantly from the background model are marked as foreground

Shadow Detection: Advanced algorithms can also detect and handle shadows separately

Key Algorithms Used
Algorithm	Description	Strengths	Weaknesses
MOG2 (Mixture of Gaussians)	Uses Gaussian mixture models for each pixel	Good for varying lighting, handles shadows	More computationally expensive
KNN (K-Nearest Neighbors)	Uses KNN classification for pixel classification	Faster, handles dynamic backgrounds well	May miss some foreground details
Complete Code
python
#                   -----video background removing-----

# SOME THEORY ABOUT THE BACKGROUND REMOVAL IN THE VIDEO..

# Background substraction is a way to access the foreground objects.
# Technically you need to extract moving foreground from static background.
# There are multiple approaches for multiple background removal.
# We discuss all of them.

import cv2
import numpy as np

# ---------- Loading Video (use raw string r"" for path) ----------
cap = cv2.VideoCapture(r"A:\computer_Vision\2008.mp4")

# Old_algo = cv2.bgsegm.createBackgroundSubtractorMOG()  # Deprecated
algo1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
algo2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ---------- Starting Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 300))
    res1 = algo1.apply(frame)
    res2 = algo2.apply(frame)
    
    # ---------- Display Side ----------
    cv2.imshow("Video", frame)
    cv2.imshow("result1 (MOG2):", res1)
    cv2.imshow("result2 (KNN):", res2)

    # Press ESC to exit
    if cv2.waitKey(25) & 0xFF == 110:  # press >>> 'n' <<< to exit
        break

cap.release()
cv2.destroyAllWindows()
Code Breakdown
1. Import Libraries
python
import cv2
import numpy as np
OpenCV: For video processing and background subtraction

NumPy: For array operations (though not explicitly used here)

2. Video Loading
python
cap = cv2.VideoCapture(r"A:\computer_Vision\2008.mp4")
Uses raw string (r"") to handle Windows file paths with backslashes

Returns a VideoCapture object for frame-by-frame reading

3. Background Subtractor Initialization
MOG2 Algorithm
python
algo1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
MOG2: Mixture of Gaussians version 2

Uses Gaussian Mixture Models to model each pixel's background

detectShadows=True: Enables shadow detection (shadows appear as gray)

KNN Algorithm
python
algo2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)
KNN: K-Nearest Neighbors approach

Uses distance-based classification for pixel grouping

Faster than MOG2 for real-time applications

4. Video Processing Loop
python
while True:
    ret, frame = cap.read()
    if not ret:
        break
Reads frames one by one

ret: Boolean indicating successful read

frame: The actual image frame

5. Frame Processing
python
frame = cv2.resize(frame, (500, 300))
res1 = algo1.apply(frame)
res2 = algo2.apply(frame)
Resizes frames for consistent display and faster processing

Applies both background subtraction algorithms

Returns binary masks where foreground is white (255), background is black (0)

6. Display Results
python
cv2.imshow("Video", frame)
cv2.imshow("result1 (MOG2):", res1)
cv2.imshow("result2 (KNN):", res2)
Shows original video and both processed results side by side

Each in a separate window

7. Exit Control
python
if cv2.waitKey(25) & 0xFF == 110:  # press 'n' to exit
    break
Waits 25ms between frames

Checks if 'n' key is pressed (ASCII value 110)

Break out of loop when key is pressed

8. Cleanup
python
cap.release()
cv2.destroyAllWindows()
Releases video capture resource

Closes all OpenCV windows

Visual Representation
Output Windows
text
+---------------------------+  +---------------------------+
|       ORIGINAL VIDEO      |  |   MOG2 RESULT (res1)     |
|                           |  |                           |
|   🚶‍♂️ Person walking     |  |   🚶‍♂️ (White foreground)  |
|   Background scene       |  |   Background (Black)     |
|                           |  |   Shadows (Gray)         |
+---------------------------+  +---------------------------+

+---------------------------+
|   KNN RESULT (res2)      |
|                           |
|   🚶‍♂️ (White foreground)  |
|   Background (Black)     |
|   Shadows (Gray)         |
+---------------------------+
Pixel Classification
text
Original Frame          →    MOG2/KNN Apply
[Background + Object]   →    [Foreground Mask]

       RGB Image                    Binary Image
   (Color information)         (White = Moving Object
                                   Black = Background)
Key Parameters & Customization
MOG2 Parameters
python
algo1 = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Number of frames for background modeling
    varThreshold=36,       # Threshold for pixel classification
    detectShadows=True     # Detect shadows
)
KNN Parameters
python
algo2 = cv2.createBackgroundSubtractorKNN(
    history=500,           # Number of frames for background modeling
    dist2Threshold=400.0,  # Distance threshold for classification
    detectShadows=True     # Detect shadows
)
Common Parameter Adjustments
Parameter	Effect	Use Case
history	More frames = better background model	Static scenes: higher value
varThreshold	Lower = more sensitive to changes	Noisy videos: increase value
detectShadows	True = shadows gray; False = shadows as foreground	Remove shadows if not needed
How to Run the Code
Install Dependencies:

bash
pip install opencv-python opencv-contrib-python numpy
Prepare Video:

Place your video at the specified path

Or change the path to your own video file

Run the Script:

bash
python 50-video_background_removal.py
Controls:

Press 'n' key to exit

The video will loop until you press 'n'

Common Issues & Solutions
Issue	Solution
Video not loading	Check file path and format. Use r"path" for Windows
Slow performance	Reduce frame size in cv2.resize()
Too much noise	Adjust varThreshold or dist2Threshold
Shadows appearing as foreground	Set detectShadows=False if not needed
Memory error	Reduce history parameter value
Enhanced Version (With Saving Output)
python
# Save processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (500, 300))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (500, 300))
    res1 = algo1.apply(frame)
    
    # Convert grayscale mask to 3-channel for saving
    res1_color = cv2.cvtColor(res1, cv2.COLOR_GRAY2BGR)
    out.write(res1_color)
    
    cv2.imshow('Processed', res1)

out.release()


```python
#                   -----video background removing-----

#SOME THEORY ABOUT THE BACKGROUND REMOVAL IN THE VIDEO..

# Background substraction is a way to access the foreground objects.
# Technically you need to extract moving foreground from static background.
# There are mulyiple approach for multiple background removal.
# We discuss all of them.



import cv2
import numpy as np

# ---------- Loading Video (use raw string r"" for path) ----------
cap = cv2.VideoCapture(r"A:\computer_Vision\2008.mp4")

#Old_algo = cv2.bgsegm.createBackgorundSubtractorMOG()
algo1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
algo2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)


# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ---------- Starting Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 300))
    res1 = algo1.apply(frame)
    res2 = algo2.apply(frame)
    
#----------Display Side----------

    cv2.imshow("Video", frame)
    cv2.imshow("result1:",res1)
    cv2.imshow("result2:",res2)

    # Press ESC to exit
    if cv2.waitKey(25) & 0xFF == 110:  #press >>>n<<< to exit...
        break

cap.release()
cv2.destroyAllWindows()
```

                                    [THE END]
