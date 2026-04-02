# Python Basics to Advance complete course
# This github helps in learning basics of python basics and its advance concept
---

## 📌 Table of Contents

* [3D Point Cloud](#3d-point-cloud)
* [Voxel Downsampling](#voxel-downsampling)
* [Outlier Removal](#outlier-removal)
* [KD-Tree](#kd-tree)
* [3D Mesh](#3d-mesh)
* [Mesh Operations](#mesh-operations)
* [Sampling](#sampling)
* [RGBD Handling](#rgbd-handling)
* [Voxelization](#voxelization)
* [Octree](#octree)
* [Surface Reconstruction](#surface-reconstruction)
* [Transformations](#transformations)
* [Mesh Deformation](#mesh-deformation)
* [Intrinsic Shape Signatures](#intrinsic-shape-signatures)
* [Ray Casting](#ray-casting)
* [Registration (ICP)](#registration-icp)
* [Visualization](#visualization)
* [Web Visualizer](#web-visualizer)
* [Open3D for TensorBoard](#open3d-for-tensorboard)
* [Built‑in Datasets](#built-in-datasets)
* [Important Techniques](#important-techniques)

---

# 🟦 Face and eyes Detection On Image.



# Introduction


So today we read about Face detection in an image is a computer vision technique used to locate and identify human faces within a picture. It works by analyzing visual features such as edges, shapes, and patterns that resemble facial structures.

In OpenCV, face detection is commonly performed using Haar Cascade classifiers. This technique is widely used in security systems, cameras, and biometric applications.


# Here we starts some questions


**Q No 1**  What is the ‘Haarcascade file’?


**Ans** The ‘HaarCascade file’ is a classifier file in which it is defined what a face looks like, including its parameters and features. All possible details about the face are available in that file. It is a part of computer vision.


**Q No 2** What is the code of loading an image?


**Ans** Here is the some code to loading an image>


**Code Input**

import cv2


import numpy as np


**Load Image**

 
image = cv2.imread(r"A:\computer_Vision\56.jpg")


**Display Image**

 
cv2.imshow("Original Image", image)


cv2.waitKey(0)


cv2.destroyAllWindows()

	 	
**Qno3** What is the code of converting an image in the gray scale?


**Ans** Here is the full code of converting an image into gray scale>:


**Code Input**


import cv2


import numpy as np


**Load Image**


image = cv2.imread(r"A:\computer_Vision\56.jpg")


**Convert to Gray**


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


**Display Images**
cv2.imshow("Original Image", image)
cv2.imshow("Gray Scale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

	 
**Qno4** Write full code of detecting faces in an image?


**Ans** Here is the full code of detecting faces in an image >


**Input Code**


import cv2
import numpy as np


**Load Image**
image = cv2.imread(r"A:\computer_Vision\56.jpg")
if image is None:
print("Error: Image not found. Check file path.")
    exit()

	
 **Resize Image**
image = cv2.resize(image, (400, 400))


 **Convert to Grayscale**
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


 **Load Haar Cascade**
face = cv2.CascadeClassifier(
 cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

 
 **Detect Faces**
faces = face.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(20, 20))

	
 **Copy image for face detection result**
face_img = image.copy()


**Draw Rectangles**
for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 2)


**Display Images**
cv2.imshow("Original Image :", image)
cv2.imshow("Gray Image :", gray)
cv2.imshow("Face Detected Image :", face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

	 
**Qno5** What is the full code of detecting eyes and faces?


**Ans** Here is the full code of detecting faces and eyes>


## Sample code 

```python
import cv2
import numpy as np


			*Load Image*
image_path = r"A:\computer_Vision\56.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()
			*Resize Image*
image = cv2.resize(image, (500, 500))  # Resize to 500x500


			*Convert to Gray*
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


			*Load Haar Cascades*
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


			*Detect Faces and Eyes*
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,  *More sensitive for faces*
    minNeighbors=4,
    minSize=(30, 30)
)


			*through all faces*
for (x, y, w, h) in faces:
    **Draw rectangle around face (Blue) with thin border**
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)    


			*Region of interest for eyes*
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    
    *Detect Eyes inside this face:*
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.03,  *Even smaller step for more accuracy*
        minNeighbors=2,     *Lower to detect additional eyes*
        minSize=(8, 8)      *Smaller size to catch tiny eyes)*


    *Loop through all detected eyes*
    for (ex, ey, ew, eh) in eyes:


        *Draw rectangle around eyes (Pink) with thin border.*
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 255), 1)


			*Display Image*
cv2.imshow("roi:",roi_color)
cv2.imshow("Face and Eyes Detection:", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
---

![Alt Text](images/one.jpg)




# Drawing Functions in OpenCV

## Introduction
Drawing functions in OpenCV are used to create shapes on images or video frames.  
They help to draw lines, rectangles, circles, ellipses, and text.

These are useful for:
- Highlighting objects  
- Marking regions  
- Visualizing results  

---

## Q1 What are Drawing Functions?
Drawing functions are used to draw shapes and text on images.  
They modify the image directly.

# Sample Of Code.
```python
import cv2
import numpy as np


img = cv2.imread("image.jpg")
img = cv2.resize(img, (500, 500))


cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


**performing the simple line code**
img = cv2.line(img, (0, 0), (200, 200), (255, 55, 255), 2)


**Drawing the arrowed line**
img = cv2.arrowedLine(img, (0, 350), (255, 255), (255, 50, 200), 2)


**Drawing the rectangle**
img = cv2.rectangle(img, (384, 10), (620, 150), (255, 250, 200), 5)


**Drawing a circle**
img = cv2.circle(img, (270, 145), 100, (230, 130, 240), 2)


**Now putting the text**
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
img = cv2.putText(
    img,
    "SHAMEER...",
    (20, 500),
    font,
    2,
    (130, 255, 110),
    2,
    cv2.LINE_AA
)


**Drawing the ellips**
img = cv2.ellipse(img, (320, 240), (110, 140), 0, 0, 360, (255, 226, 120), 4)


**Creating a full black image**
import numpy as np
img = np.zeros((512, 512, 3), np.uint8)


**Creating a full white image**
img = np.ones((512, 512, 3), np.uint8) * 255


**Last lines of code which are the most importat**
cv2.waitKey(0)
cv2.desttroyeAllWindows.
```
#![Alt Text](images/602.jpg)



# Removing Background in an Image using OpenCV

## Code

```python
import cv2
import numpy as np


		**Load the main image**
img = cv2.imread(r"A:\computer_Vision\920.jpg")


**Resizing the image**
img = cv2.resize(img, (400, 400))


		**Convert image to HSV**
hsv_original = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


		**Load ROI image**
roi = cv2.imread(r"A:\computer_Vision\bgr.jpg")
roi = cv2.resize(roi, (100, 100))
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


**Create histogram of ROI**
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Backprojection to create mask
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# Filter and remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

# Merge mask with original image
mask_3ch = cv2.merge([mask, mask, mask])
result = cv2.bitwise_and(img, mask_3ch)

# Display results
cv2.imshow("Original Image", img)
cv2.imshow("HSV Original Image", hsv_original)
cv2.imshow("ROI Image", roi)
cv2.imshow("HSV ROI Image", hsv_roi)
cv2.imshow("Mask Image", mask)
cv2.imshow("Result Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

#uotput result:
#![Alt Text](images/902.jpg)




# Bitwise Operations in OpenCV

## Introduction
Bitwise operations include AND, OR, NOT, and XOR.  
They are used for tasks like masking and finding regions of interest (ROI).

---

## Code

```python
import cv2
import numpy as np


**Create blank images
img1 = np.zeros((250, 500, 3), np.uint8)
img2 = np.zeros((250, 500, 3), np.uint8)


**Draw rectangles
img1 = cv2.rectangle(img1, (150, 100), (200, 250), (255, 255, 255), -1)
img2 = cv2.rectangle(img2, (10, 10), (170, 190), (255, 255, 255), -1)


**Show images
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)


**AND operation
bitAnd = cv2.bitwise_and(img1, img2)
cv2.imshow("bitAnd", bitAnd)

**OR operation
bitOr = cv2.bitwise_or(img1, img2)
cv2.imshow("bitOr", bitOr)


**NOT operation
bitNot1 = cv2.bitwise_not(img1)
bitNot2 = cv2.bitwise_not(img2)
cv2.imshow("bitNot1", bitNot1)
cv2.imshow("bitNot2", bitNot2)


**XOR operation
bitXor = cv2.bitwise_xor(img1, img2)
cv2.imshow("bitXor", bitXor)


cv2.waitKey(0)
cv2.destroyAllWindows()
```


# Contours and its Functions in OpenCV


**There Are Two Methods**:


## *Introduction*
Contours are used to detect shapes in images.  
Main functions:
- Moments  
- Approximation  
- Convex Hull  
---


***Method No 1***:


*Code Of sample*:

```python:
import cv2
import numpy as np

Load image:
img = cv2.imread("shapes.png")
img = cv2.resize(img, (250, 250))

Convert to grayscale:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Apply threshold:
ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

Find contours:
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours:", len(cnts))

Loop through contours:
for c in cnts:
    M = cv2.moments(c)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        	Area:
        area = cv2.contourArea(c)

        	:Approximation
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        	Convex Hull:
        hull = cv2.convexHull(approx)

        	Bounding box:
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(img, (x, y), (x+w, y+h), (125, 10, 20), 1)

        Draw center:
        cv2.circle(img, (cX, cY), 3, (222, 222, 22), -1)
        cv2.putText(img, "Center", (cX - 20, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

Display:
cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", gray)
cv2.imshow("Threshold Image", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
```


###Method No 2:


# Approximation and Convex Hull in OpenCV

***Sample Of Code***

```python
import cv2
import numpy as np

	Load image:
img = cv2.imread("shapes.png")
img = cv2.resize(img, (250, 250))

	Convert to grayscale:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	Apply threshold:
ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

	Find contours:
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours:", len(cnts))

area1 = []

	Loop through contours:
for c in cnts:
    M = cv2.moments(c)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        	Area:
        area = cv2.contourArea(c)
        area1.append(area)

        	Approximation:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        	Convex Hull:
        hull = cv2.convexHull(approx)

        	Bounding rectangle:
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(img, (x, y), (x+w, y+h), (125, 10, 20), 1)

        	Draw center:
        cv2.circle(img, (cX, cY), 3, (222, 222, 22), -1)
        cv2.putText(img, "Center", (cX - 20, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

	Display images:
cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", gray)
cv2.imshow("Threshold Image", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()



