# =============================================================
# Script to segment frames for all video directories in the "data" directory.
# =============================================================
# Configure Script
imgExtension = ".jpg"
MIN_EYE_SIZE = 40 # Minimum number of pixels to be contained in the segmentation of an eye
dataPath = "/workspace/data" # Path to where the videos are housed
modelPath = "/workspace/utils/iTrackerModelFiles" # Path to iTracker model files
myUtilsPath = '/workspace/utils'
classifierFilesDir = "/workspace/utils/opencv_haarcascades"
# ================================================
# Libraries
import numpy as np
import cv2
import pandas as pd
import os
import shutil
import matplotlib as mpl
mpl.use('Agg')
import skimage.io as io
# =============================================================
# Function to generate the faceGrid
def GenerateFaceGrid(faceXTopLeft, faceYTopLeft, faceW, faceH, frameW, frameH, gridSize=25):
    scaleX = gridSize / float(frameW)
    scaleY = gridSize / float(frameH)

    # Scale the image to gridSize x gridSize
    xLow = round(faceXTopLeft * scaleX) + 1
    yLow = round(faceYTopLeft * scaleY) + 1
    w = round(faceW * scaleX)
    h = round(faceH * scaleY)

    # Generate the facegrid
    faceGrid = np.zeros((gridSize, gridSize))
    xHi = xLow + w - 1
    yHi = yLow + h - 1
    #     print yLow,yHi,xLow,xHi,w,h

    # Make sure everything stays within bounds
    xLow = int(min(gridSize, max(1, xLow)))
    yLow = int(min(gridSize, max(1, yLow)))
    xHi = int(min(gridSize, max(1, xHi)))
    yHi = int(min(gridSize, max(1, yHi)))

    faceGrid[yLow:yHi, xLow:xHi] = 1

    return faceGrid
# ================================================
# Lists with different weights to try for the Viola Jones algorithm
faceClassifierList = ['haarcascade_frontalface_default.xml',
                      'haarcascade_frontalface_alt.xml',
                      'haarcascade_frontalface_alt2.xml',
                      'haarcascade_frontalface_alt_tree.xml',
                      'haarcascade_profileface.xml']
eyeClassifierList = ['haarcascade_eye.xml',
                     'haarcascade_eye_tree_eyeglasses.xml']

faceClassifierList = [classifierFilesDir + "/" + classifierName for classifierName in faceClassifierList]
eyeClassifierList = [classifierFilesDir + "/" + classifierName for classifierName in eyeClassifierList]

# =============================================================
# Main Part of the Program
# =============================================================
videoNameList = [fName for fName in os.listdir(dataPath) if os.path.isdir(dataPath+"/"+fName)]

for videoName in videoNameList:
    print("------------------------------------")
    print("Processing files for the video: %s" % (videoName))
    # Set up folders to hold segmentations
    currFrameDir = dataPath + "/" + videoName + "/frames"
    leftEyeDir = dataPath + "/" + videoName + "/leftEye/"
    rightEyeDir = dataPath + "/" + videoName + "/rightEye/"
    faceDir = dataPath + "/" + videoName + "/face/"
    faceGridDir = dataPath + "/" + videoName + "/faceGrid/"
    notSegmentedDir = dataPath + "/" + videoName + "/notSegmented/"
    if not os.path.exists(leftEyeDir):
        os.makedirs(leftEyeDir)
    if not os.path.exists(rightEyeDir):
        os.makedirs(rightEyeDir)
    if not os.path.exists(faceDir):
        os.makedirs(faceDir)
    if not os.path.exists(faceGridDir):
        os.makedirs(faceGridDir)
    if os.path.exists(notSegmentedDir):
        shutil.rmtree(notSegmentedDir)
    os.makedirs(notSegmentedDir)

    # Segment the images
    nNotSegmented = 0
    frameList = [int(filter(str.isdigit, fName)) for fName in os.listdir(currFrameDir) if ".jpg" in fName]
    nFrames = len(frameList)
    # Set up data frame to hold information about the segmentations
    index = range(nFrames)
    columns = ['FrameId', 'FacePosX', 'FacePosY', 'FaceW', 'FaceH', 'LeftEyeW', 'LeftEyeH', 'RightEyeW', 'RightEyeH']
    frameCharDf = pd.DataFrame(index=index, columns=columns)
    frameCharDf = frameCharDf.fillna(0)  # with 0s rather than NaNs
    for imgId in frameList:
        # Load the image
        img = cv2.imread(currFrameDir + "/img" + str(imgId) + imgExtension, 1)

        # Convert to greyscale for processing (Note: openCV uses BGR rather than RGB)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # -- Extract face and eyes using the Viola Jones algorithm -----
        # Try all weights available with openCV until find a face
        scaleFactor = 1.1
        maxScaleFactor = 7
        keepSearchingB = True
        while keepSearchingB:
            for classifierName in faceClassifierList:
                # Load the weights
                face_cascade = cv2.CascadeClassifier(classifierName)
                faces = face_cascade.detectMultiScale(grey, scaleFactor, 2, minSize=(100,100))
                if len(faces) == 1: break
            if len(faces) > 0 or scaleFactor > maxScaleFactor:
                keepSearchingB = False
                if len(faces) == 1: break
            else:
                print("Change Face Scale Factor to: %1.2f" % scaleFactor)
                scaleFactor += 0.1

        for (x, y, w, h) in faces:
            roi_color = img[y:y + h, x:x + w]
            roi_gray = grey[y:y + h, x:x + w]
            # Detect eyes. Try all classifiers available in openCV until find eyes.
            scaleFactor = 1.1
            maxScaleFactor = 7
            keepSearchingB = True
            while keepSearchingB:
                for classifierName in eyeClassifierList:
                    # Load the weights
                    eye_cascade = cv2.CascadeClassifier(classifierName)
                    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,2)
                    if len(eyes) > 1: break
                if len(eyes) > 1 or scaleFactor > maxScaleFactor:
                    keepSearchingB = False
                    if len(eyes) > 1: break
                else:
                    print("Change Eye Scale Factor to: %1.2f" % scaleFactor)
                    scaleFactor += 0.1

        # Extract the face and eyes
        if len(faces) == 1 and len(eyes) > 1:
            # Extract the face
            x, y, w, h = faces[0, :]
            faceImg = img[y:y + h, x:x + w]

            # Extract the eyes
            # Check that segmentations have a minimum
            # size (to remove nose holes and ears which are occasionally picked up)
            eyeSegSizes = zip(eyes[:, 2], eyes[:, 3])
            segToDeleteList = []
            for i, (ew, eh) in enumerate(eyeSegSizes):
                if ew < MIN_EYE_SIZE or eh < MIN_EYE_SIZE:
                    segToDeleteList = segToDeleteList + [i]
            mask = np.ones(len(eyes), dtype=bool)
            mask[segToDeleteList] = False
            eyes = eyes[mask, :]

            # If still have more than 3 segments remove the smallest segment
            while len(eyes) > 2:
                eyeSegSizes = zip(eyes[:, 2], eyes[:, 3])
                smallestSegArea = 999999
                smallestSegId = 0
                for i, (ew, eh) in enumerate(eyeSegSizes):
                    if ew * eh < smallestSegArea:
                        smallestSegId = i
                        smallestSegArea = ew * eh
                mask = np.ones(len(eyes), dtype=bool)
                mask[smallestSegId] = False
                eyes = eyes[mask, :]

            # Extract eyes
            if len(eyes) == 2:
                # Find the right eye by looking for the segmentation further
                # to the left in the image (smaller ex)
                rightEyeIdx = np.argmin(eyes[:, 0])
                leftEyeIdx = (rightEyeIdx + 1) % 2
                ex, ey, ew, eh = eyes[rightEyeIdx, :]
                rightEyeImg = roi_color[ey:ey + eh, ex:ex + ew]
                ex, ey, ew, eh = eyes[leftEyeIdx, :]
                leftEyeImg = roi_color[ey:ey + eh, ex:ex + ew]
            else:
                print "Couldn't segment img_" + str(imgId)
                print "Because detected " + str(len(eyes)) + " eye segments:"
                print eyes
                nNotSegmented += 1
                io.imsave(notSegmentedDir + str(imgId) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                continue

            # Generate the facegrid
            faceGridImg = GenerateFaceGrid(faces[0][0], faces[0][1], faces[0][2], faces[0][3], img.shape[1],
                                           img.shape[0])

            # Convert everything to RGB for saving
            faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB)
            leftEyeImg = cv2.cvtColor(leftEyeImg, cv2.COLOR_BGR2RGB)
            rightEyeImg = cv2.cvtColor(rightEyeImg, cv2.COLOR_BGR2RGB)

            # Save the images
            io.imsave(leftEyeDir + str(imgId) + ".jpg", leftEyeImg)
            io.imsave(rightEyeDir + str(imgId) + ".jpg", rightEyeImg)
            io.imsave(faceDir + str(imgId) + ".jpg", faceImg)
            io.imsave(faceGridDir + str(imgId) + ".jpg", faceGridImg)

            # Save the characteristics
            frameCharDf.loc[imgId] = [imgId, faces[0][0], faces[0][1], faces[0][2], faces[0][3],
                                      eyes[leftEyeIdx, 2], eyes[leftEyeIdx, 3],
                                      eyes[rightEyeIdx, 2], eyes[rightEyeIdx, 3]]
        else:
            if len(faces) == 0:
                print "Segmentation failed: OpenCV couldn't detect a face in img_" + str(imgId)
            elif len(eyes) == 0:
                print "Segmentation failed: OpenCV couldn't detect eyes in img_" + str(imgId)
            else:
                print "Segmentation failed in img_" + str(imgId) + ". Not clear why. Here are details: "
                print "Number of faces regions detected: " + str(len(faces))
                print "Number of eye regions detected: " + str(len(eyes))
            nNotSegmented += 1
            io.imsave(notSegmentedDir + str(imgId) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print("Segmented %i of %i frames."%(nFrames-nNotSegmented,nFrames))

print("Done.")
print("------------------------------------")
