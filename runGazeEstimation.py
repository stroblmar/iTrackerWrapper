# ================================================
# Script to estimate gaze all video directories in the "data" directory.
# =============================================================
# Configure Script
logInterval = 100 # Interval (in frames) at which progress update is printed to the screen

# Set directories
dataPath = "/workspace/iTrackerWrapper/data" # Path to where the videos are housed
modelPath = "/workspace/utils/iTrackerModelFiles" # Path to iTracker model files
caffePath = '/opt/caffe'  # path to caffe
# ================================================
# Libraries
import scipy.io
import os
import sys
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
sys.path.insert(0, caffePath + '/python')
import caffe
sys.path.insert(0, ".")
import iTrackerWrapper as itw
# =============================================================
# Main Part of the Program
# =============================================================
# Set up the net
model_def = modelPath+'/itracker_deploy_BatchSize1.prototxt'
model_weights = modelPath+'/snapshots/itracker_iter_92000.caffemodel'

gazeEstimator = itw.GazeEstimator(model_def,model_weights,dataPath)

# Set up the data pre-processor
# Load the mean gazeCapture image (as distributed with GazeCapture) for normalisation
meanFace = scipy.io.loadmat(modelPath+'/mean_images/mean_face_224.mat')
meanFace = meanFace['image_mean'].mean(axis=(0, 1))
meanLeftEye = scipy.io.loadmat(modelPath+'/mean_images/mean_left_224.mat')
meanLeftEye = meanLeftEye['image_mean'].mean(axis=(0, 1))
meanRightEye = scipy.io.loadmat(modelPath+'/mean_images/mean_right_224.mat')
meanRightEye = meanRightEye['image_mean'].mean(axis=(0, 1))

gazeEstimator.SetupPreProcessor(meanLeftEye,meanRightEye,meanFace)
# --------------------------------------------------------------
# Process the images
videoNameList = [fName for fName in os.listdir(dataPath) if os.path.isdir(dataPath+"/"+fName)]

for videoName in videoNameList:
    print("------------------------------------")
    print("Processing files for the video: %s" % (videoName))
    currFrameDir = dataPath + "/" + videoName + "/face"
    frameList = [int(filter(str.isdigit, fName)) for fName in os.listdir(currFrameDir) if ".jpg" in fName]

    # Set up pandas dataframe to hold results
    index = range(len(frameList))
    columns = ['VideoName', 'FrameId', 'EstPosX', 'EstPosY'] #+ ["Activation"+str(i) for i in range(128)]
    performanceDF = pd.DataFrame(index=index, columns=columns)
    performanceDF = performanceDF.fillna(0)  # with 0s rather than NaNs

    dfIdx = 0  # Index of current row in result data frame
    for i, frameId in enumerate(frameList):
        if i % logInterval == 0:
            print "Processing frame: " + str(i) + " of " + str(len(frameList))

        # Load the data
        # Set up the paths
        subjectDir = dataPath + "/" + videoName
        imgName = str(frameId) + '.jpg'
        leftEyeImgPath = subjectDir + '/leftEye/' + imgName
        rightEyeImgPath = subjectDir + '/rightEye/' + imgName
        faceImgPath = subjectDir + '/face/' + imgName
        faceGridImgPath = subjectDir + '/faceGrid/' + imgName

        # Load the images
        leftEyeImg = caffe.io.load_image(leftEyeImgPath)
        rightEyeImg = caffe.io.load_image(rightEyeImgPath)
        faceImg = caffe.io.load_image(faceImgPath)
        faceGridImg = caffe.io.load_image(faceGridImgPath)
        faceGridImg = faceGridImg[:, :, 0]
        currFrameDataDict = {'leftEyeImg': leftEyeImg, 'rightEyeImg': rightEyeImg, 'faceImg': faceImg, 'faceGridImg': faceGridImg}

        # Estimate Gaze
        predPosVec = gazeEstimator.EstimateGazeForImage(currFrameDataDict)
        performanceDF.loc[dfIdx] = [videoName, frameId, predPosVec[0], predPosVec[1]] #+ gazeEstimator.net.blobs['fc1'].data.tolist()[0]
        dfIdx += 1

    outName = dataPath + "/" + videoName + "/" + "gazePredictions.csv"
    performanceDF.to_csv(outName, sep=',', index=False)

print "Done."
print("==========================================")

