#           ----------Object tracking and detecting----------

#Some Theroy About The Object Tracking And Detecting):-

# Object tracking using Meanshoft algo.
# The idea behind this algo is to move small window to get high density pixels
# same as histograme background projection.
# Here are the some steps to uses the algo):-
# step no1):- Target and find its histogram for backproject the target.
# step no2):- Algo sets one initial location.
# step no3):- Setup the termination critaria.


import cv2
import numpy as np

# ----------Load video----------
cap = cv2.VideoCapture(r"A:\computer_Vision\2008.mp4")
if not cap.isOpened():
    print("Error: Video not found")
    exit()

# ----------Initialize HOG human detector----------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ----------Random ROI settings----------
width, height = 120, 180
roi_frozen = False
frozen_roi = None

# ----------Select initial random ROI----------
ret, frame = cap.read()
if ret:
    frame = cv2.resize(frame, (500, 300))
    h, w, _ = frame.shape
    x = np.random.randint(0, w - width)
    y = np.random.randint(0, h - height)
    frozen_roi = frame[y:y + height, x:x + width]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 300))
    output_frame = frame.copy()

    # ----------Detect humans----------
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
    for (hx, hy, hw, hh) in boxes:
        cv2.rectangle(output_frame, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 2)
        cv2.putText(output_frame, "Person", (hx, hy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # ----------Draw rectangle around frozen ROI----------
    cv2.rectangle(output_frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # ----------Show video and frozen ROI----------
    cv2.imshow("Video with Humans", output_frame)
    if frozen_roi is not None:
        cv2.imshow("Frozen ROI", frozen_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 32:  # SPACE to freeze ROI permanently
        roi_frozen = True
    if k == 110:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# ----------Load video----------
cap = cv2.VideoCapture(r"A:\computer_Vision\2008.mp4")
if not cap.isOpened():
    print("Error: Video not found")
    exit()

# ----------HOG Human Detector----------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 450))

    # ----------Detect humans----------
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    all_pts = []

    for (x, y, w, h) in boxes:
        # Convert all values to float (IMPORTANT FIX)
        cx = float(x + w / 2)
        cy = float(y + h / 2)
        ww = float(w)
        hh = float(h)

        rect = ((cx, cy), (ww, hh), 0.0)

        box = cv2.boxPoints(rect)   # Now works correctly
        pts = np.int64(box)         # Save as int64
        all_pts.append(pts)

    # Draw all rotated rectangles
    if len(all_pts) > 0:
        final = cv2.polylines(frame, all_pts, True, (255, 0, 0), 1)

    cv2.imshow("Human Detection frames:", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('n'):
        break

cap.release()
cv2.destroyAllWindows()
