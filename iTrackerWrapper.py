# =============================================================
# Class wrapping around iTracker. Allows easy loading of iTracker and processing of images.
# Currently supports:
# - Preprocessing using the caffe tools
# - Truncation to Screen (Support for phones other than the iOS devices is currently a bit crude.
# The screen size details have to be entered manually in EstimateGazeForSubject() at xxx 'Use details for a different phone by entering them here manually xxx').
# - Multithreading
### NOTE: In order to get this to run, have to adjust caffePath and myUtilsPath to the relevant directory.
# =============================================================
caffePath = '/opt/caffe'  # path to caffe
myUtilsPath = '/workspace/utils'
# ------------------------------------------------------------
# Import libraries
import os
import scipy.misc
import numpy as np
import json
import multiprocessing as mp
import pandas as pd
import sys
import shutil
# ------------------------------------------------------------
# Import caffe
import matplotlib as mpl
mpl.use('Agg')
sys.path.insert(0, caffePath + '/python')
import caffe
# ------------------------------------------------------------
# Import utils
sys.path.insert(0, myUtilsPath)
import iTrackerUtils as utils
import multiThreadingUtils as mtUtils
# =============================================================
class GazeEstimator():
    def __init__(self,modelDefPath,modelWeightsPath,dataPath):
        # Load the net
        caffe.set_mode_cpu()
        self.net = caffe.Net(modelDefPath,modelWeightsPath,caffe.TEST)
        # Initialise other variables
        self.dataPath = dataPath
        self.tmpDir = "tmp"

    # =============================================================
    # Function to load the data
    def LoadFrame(self,subjectId, frameId, dataPath, appleFormatB=True):
        return utils.LoadFrame(subjectId, frameId, dataPath, appleFormat=appleFormatB)

    # =============================================================
    # Function to load the mean images for normalisation into class variables
    def SetupPreProcessor(self,meanLeftEye,meanRightEye,meanFace):
        self.meanLeftEye = meanLeftEye
        self.meanRightEye = meanRightEye
        self.meanFace = meanFace
        # self.PreProcessImages = preProcessFun

        # Set up the transformer
        self.transformer = caffe.io.Transformer({'leftEyeImg': self.net.blobs['image_left'].data.shape,
                                            'rightEyeImg': self.net.blobs['image_right'].data.shape,
                                            'faceImg': self.net.blobs['image_face'].data.shape})
        # Left Eye
        self.transformer.set_mean('leftEyeImg', self.meanLeftEye)  # subtract the dataset-mean value in each channel
        self.transformer.set_transpose('leftEyeImg', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_channel_swap('leftEyeImg', (2, 1, 0))  # swap channels from RGB to BGR
        self.transformer.set_input_scale('leftEyeImg', 1.0)  # rescale from [0, 1] to [0, 255]

        # Right Eye
        self.transformer.set_mean('rightEyeImg', self.meanRightEye)  # subtract the dataset-mean value in each channel
        self.transformer.set_transpose('rightEyeImg', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_channel_swap('rightEyeImg', (2, 1, 0))  # swap channels from RGB to BGR
        self.transformer.set_input_scale('rightEyeImg', 1.0)  # rescale from [0, 1] to [0, 255]

        # Face
        self.transformer.set_mean('faceImg', self.meanFace)  # subtract the dataset-mean value in each channel
        self.transformer.set_transpose('faceImg', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_channel_swap('faceImg', (2, 1, 0))  # swap channels from RGB to BGR
        self.transformer.set_input_scale('faceImg', 1.0)  # rescale from [0, 1] to [0, 255]

    # =============================================================
    # Function to preprocess the data
    def PreProcessImages(self,currFrameDataDict):
        # Resize to 224x224
        leftEyeImg224 = scipy.misc.imresize(currFrameDataDict['leftEyeImg'], (224, 224), interp='bicubic')
        rightEyeImg224 = scipy.misc.imresize(currFrameDataDict['rightEyeImg'], (224, 224), interp='bicubic')
        faceImg224 = scipy.misc.imresize(currFrameDataDict['faceImg'], (224, 224), interp='bicubic')

        # Normalise and rearrange colour channels
        leftEyeImg_Transformed = self.transformer.preprocess('leftEyeImg', leftEyeImg224)
        rightEyeImg_Transformed = self.transformer.preprocess('rightEyeImg', rightEyeImg224)
        faceImg_Transformed = self.transformer.preprocess('faceImg', faceImg224)
        faceGridImg_Transformed = np.round(currFrameDataDict['faceGridImg'].reshape((625, 1, 1)))

        return {'leftEyeImg_Transformed': leftEyeImg_Transformed,
                'rightEyeImg_Transformed': rightEyeImg_Transformed,
                'faceImg_Transformed': faceImg_Transformed,
                'faceGridImg_Transformed': faceGridImg_Transformed}

    # =============================================================
    # Function to perform gaze estimation on a given image
    def EstimateGazeForImage(self,currFrameDataDict):
        # Preprocess the image
        currFrameDataDict_Transformed = self.PreProcessImages(currFrameDataDict)

        # Load images into the net
        self.net.blobs['image_left'].data[...] = currFrameDataDict_Transformed['leftEyeImg_Transformed']
        self.net.blobs['image_right'].data[...] = currFrameDataDict_Transformed['rightEyeImg_Transformed']
        self.net.blobs['image_face'].data[...] = currFrameDataDict_Transformed['faceImg_Transformed']
        self.net.blobs['facegrid'].data[...] = currFrameDataDict_Transformed['faceGridImg_Transformed']

        # Predict the point of gaze
        output = self.net.forward()

        return np.copy(output['fc3'][0])

    # =============================================================
    # Function to perform gaze estimation on a given subject
    def EstimateGazeForSubject(self, subjectId, multiThreadedB=False,
                               computeErrorB=True, truncate2ScreenB=False,
                               appleFormatB=True, deviceInfoDFPath = "",
                               verboseLevel=1, logInterval=100, writeToFileB=False):
        if verboseLevel>=1 and multiThreadedB:
            print("------------------------------------------")
            print("Processing Subject: " + str(subjectId)) + " as process " + str(os.getpid())
            print("------------------------------------------")

        # Load a list with all the frameIds for this subject
        frameList = utils.GetFrameIds(subjectId, self.dataPath, appleFormat=appleFormatB)

        # Set up pandas dataframe to hold results
        index = range(len(frameList))
        if computeErrorB: columns = ['SubjectId', 'FrameId', 'Error', 'EstPosX', 'EstPosY',
                                     'TruePosX', 'TruePosY']
        else: columns = ['SubjectId', 'FrameId', 'EstPosX', 'EstPosY']
        performanceDF = pd.DataFrame(index=index, columns=columns)
        performanceDF = performanceDF.fillna(0)  # with 0s rather than NaNs

        if computeErrorB:
            # Load true dot positions
            dotFName = self.dataPath + "/" + str(subjectId) + "/dotInfo.json"
            with open(dotFName) as json_data:
                d = json.load(json_data)
            xPosVec = d['XCam']
            yPosVec = d['YCam']
            truePosList = np.array([xPosVec, yPosVec]).transpose()

        if truncate2ScreenB:
            if appleFormatB:
                # Load screen orientation and device info for iOS devices
                screenFName = self.dataPath + "/" + str(subjectId) + "/screen.json"
                with open(screenFName) as json_data:
                    screenInfoDict = json.load(json_data)
                orientationVec = screenInfoDict['Orientation']
                deviceInfoDF = pd.read_csv(deviceInfoDFPath)
                currDeviceInfoDF = self.GetDeviceInfo(subjectId,deviceInfoDF)
            else: # xxx Use details for a different phone by entering them here manually xxx
                orientationVec = np.ones(len(frameList))
                currDeviceInfoDict = {'DeviceCameraToScreenXMm': 54,
                     'DeviceCameraToScreenYMm': 8.5,
                     'DeviceScreenWidthMm': 63,
                     'DeviceScreenHeightMm': 111}
                currDeviceInfoDF = pd.DataFrame(currDeviceInfoDict,
                                                index=[1])

        dfIdx = 0 # Index of current row in result data frame
        for i, frameId in enumerate(frameList):
            if verboseLevel>=2 and i % logInterval == 0:
                if multiThreadedB: print "Process " + str(os.getpid()) + " processing subject " + str(subjectId) + ", frame " + str(i) + " of " + str(len(frameList))
                else: print "Processing frame: " + str(i) + " of " + str(len(frameList))
            # Load the data
            currFrameDataDict = self.LoadFrame(subjectId, frameId, self.dataPath, appleFormatB=appleFormatB)
            # Estimate Gaze
            predPosVec = self.EstimateGazeForImage(currFrameDataDict)
            # Process predictions
            if truncate2ScreenB:
                # Truncate the prediction to screen. Predictions are generated in camera space, where
                # the camera is the origin (0,0). In order to determine if a prediction is off
                # the screen, I project the predictions into screen space, where they are expressed with
                # respect to the top left corner of the screen. From here it is easy to  check if the
                # prediction is on the screen or not, by comparing it to the screen width/height. If a
                # prediction is outside the screen, it is truncated at the closest screen edge. Finally, I
                # convert the truncated prediction back into camera space, because the true dot positions
                # are given in this space. All functions have been adopted from code provided by the Torralba
                # group.
                xPred_CamSp = predPosVec[0]
                yPred_CamSp = predPosVec[1]
                orientation = orientationVec[int(frameId)]
                pred_ScreenSp = self.Cam2Screen(xPred_CamSp, yPred_CamSp, orientation, currDeviceInfoDF)
                pred_trunc = self.Truncate2Screen(pred_ScreenSp[0], pred_ScreenSp[1], orientation, currDeviceInfoDF)
                predPosVec = self.Screen2Cam(pred_trunc[0], pred_trunc[1], orientation, currDeviceInfoDF)
            if computeErrorB:
                # Compute the Eulerian Error
                err = np.sqrt(np.sum(np.square(predPosVec - truePosList[int(frameId), :])))
                # Save results to the data frame
                performanceDF.loc[dfIdx] = [subjectId, frameId, err, predPosVec[0], predPosVec[1], truePosList[int(frameId), 0],
                                        truePosList[int(frameId), 1]]
            else:
                performanceDF.loc[dfIdx] = [subjectId, frameId, predPosVec[0], predPosVec[1]]
            dfIdx+=1

        # Return results
        if multiThreadedB or writeToFileB:
            outName = "tmp/" + str(subjectId) + ".csv"
            performanceDF.to_csv(outName, sep=',', index=False)
            if multiThreadedB and verboseLevel>=1: print "Process " + str(os.getpid()) + " - Done."
        else:
            return performanceDF

    # =============================================================
    # Function to perform gaze estimation for a list of subjects. Returns a pandas data frame with the
    # results in format ['SubjectId', 'FrameId', 'Error', 'EstPosX', 'EstPosY', 'TruePosX', 'TruePosY']
    # The function can do both serial processing or multi-threading. In the latter case, the subjects
    # are distributed across different threads and processed in parallel.
    def EstimateGazeForSubjectList(self,subjectList=[],subjectsListFName="",outFName="",
                                   truncate2ScreenB=False, deviceInfoDFPath = "",
                                   multiThreadedB=True,nCores=2,computeErrorB=True,appleFormatB=True,
                                   verboseLevel=1,logInterval=100):
        # Check if list is provided as list, or by file. If it is provided by file, read in the list.
        if len(subjectList)==0 and len(subjectsListFName)>0:
            with open(subjectsListFName) as file:
                subjectList = file.read()
            subjectList = subjectList.split("\n")
            subjectList = [int(float(x)) for x in subjectList if
                           len(x) > 0]  # Little bit of fiddling to bring into integers and remove the trailing new line
            # Convert subject list into a list of 5-digit strings (i.e. go from 10 to 00010) for compatibility with the naming
            # system in gazeCapture
            subjectList = [utils.NumTo5DigitCode(subjectId) for subjectId in subjectList]
        elif len(subjectList)==0 and len(subjectsListFName)==0:
            raise NameError("No subject list to process provided.")

        # Set up a temporary folder to hold the results for the subjects that have been processed
        if not os.path.exists(self.tmpDir):
            os.makedirs(self.tmpDir)

        # Process the list by estimating the gaze for each subject
        # 1) Multi-threaded Implementation:
        if multiThreadedB:
            # Define an output queue to hold the results of the individual processes (i.e. each subject)
            jobManager = mtUtils.ThreadManager(nCores)
            # Add jobs to the queue
            for subjectId in subjectList:
                currentProcess = mp.Process(target=self.EstimateGazeForSubject,
                                            args=(subjectId,True,computeErrorB,
                                                  truncate2ScreenB,appleFormatB,
                                                  deviceInfoDFPath,verboseLevel,
                                                  logInterval,True))
                jobManager.AddJob(currentProcess)
            # Run the queue
            jobManager.RunQueue(verboseLevel=verboseLevel-2)

        # 2) Serial Implementation:
        else:
            for subjectId in subjectList:
                if verboseLevel>=1:
                    print("------------------------------------------")
                    print("Processing Subject: " + subjectId + "; Done " + str(subjectList.index(subjectId)) + " of " + str(
                        len(subjectList)))
                    print("------------------------------------------")
                    self.EstimateGazeForSubject(subjectId,
                                                truncate2ScreenB=truncate2ScreenB,
                                                verboseLevel=verboseLevel,
                                                computeErrorB=computeErrorB,
                                                appleFormatB=appleFormatB,
                                                deviceInfoDFPath=deviceInfoDFPath,
                                                logInterval=logInterval,
                                                writeToFileB=True)

        # Concatenate results, which are stored in multiple files in tmp into one data frame
        resultsDF = self.CollectResults()

        # Clean up
        # shutil.rmtree(self.tmpDir)

        # Return
        if len(outFName)>0:
            # Save performance results to file
            if verboseLevel>=1:
                print("==========================================")
                print "Saving Results"
            resultsDF.to_csv(outFName, sep=',', index=False)
            if verboseLevel>=1:
                print "Done."
                print("==========================================")
        else:
            return resultsDF

    # =============================================================
    # Function to collect the results stored in the temporary directory and return them as a single data frame.
    def CollectResults(self):
        resultList = []
        for resultFile in os.listdir(self.tmpDir):
            resultList.append(pd.read_csv(self.tmpDir+"/" + resultFile))
        return pd.concat(resultList)

    # =============================================================
    # Function to get the device info (phone Type, screen dimensions etc.)
    # for the current subject.
    def GetDeviceInfo(self,subjectId,deviceInfoDF):
        # Load phone type
        infoFName = self.dataPath + "/" + str(subjectId) + "/info.json"
        with open(infoFName) as json_data:
            expSetupDict = json.load(json_data)
        deviceName = expSetupDict['DeviceName']

        # Correct for different spellings
        if deviceName=="iPhone 5S": deviceName="iPhone 5s"
        if deviceName=="iPhone 6S": deviceName="iPhone 6s"
        if deviceName=="iPhone 5C": deviceName="iPhone 5c"
        if deviceName=="iPhone 4S": deviceName="iPhone 4s"

        return deviceInfoDF[deviceInfoDF['DeviceName'] == deviceName]

    # =============================================================
    # Function to convert positions from camera space in which the camera is the origin, to
    # screen space, in which the top left corner of the screen is the origin.
    def Cam2Screen(self,xPos_CamSp,yPos_CamSp,orientation,currDeviceInfoDF):
        # Convert input from cm to mm to be compatible with the measurements provided by Apple.
        xPos_CamSp = xPos_CamSp * 10
        yPos_CamSp = yPos_CamSp * 10

        # Transform the coordinates from camera space in which the camera is the origin, to
        # screen space, where the top left corner of the screen is the origin, and the
        # screen is the first quadrant.
        dX = float(currDeviceInfoDF['DeviceCameraToScreenXMm'])
        dY = float(currDeviceInfoDF['DeviceCameraToScreenYMm'])
        dW = float(currDeviceInfoDF['DeviceScreenWidthMm'])
        dH = float(currDeviceInfoDF['DeviceScreenHeightMm'])

        if orientation == 1:
            xPos_ScreenSp = xPos_CamSp + dX
            yPos_ScreenSp = -yPos_CamSp - dY
        elif orientation == 2:
            xPos_ScreenSp = xPos_CamSp - dX + dW
            yPos_ScreenSp = -yPos_CamSp + dY + dH
        elif orientation == 3:
            xPos_ScreenSp = xPos_CamSp - dY
            yPos_ScreenSp = -yPos_CamSp - dX + dW
        else:
            xPos_ScreenSp = xPos_CamSp + dY + dH
            yPos_ScreenSp = -yPos_CamSp + dX

        # Convert back to cm
        xPos_ScreenSp /= 10
        yPos_ScreenSp /= 10

        return [xPos_ScreenSp,yPos_ScreenSp]

    # =============================================================
    # Function to convert positions from screen space in which the top left corner of the screen is the origin, to
    # screen space, in which the camera is the origin.
    def Screen2Cam(self,xPos_ScreenSp,yPos_ScreenSp,orientation,currDeviceInfoDF):
        # Convert input from cm to mm to be compatible with the measurements provided by Apple.
        xPos_ScreenSp = xPos_ScreenSp * 10
        yPos_ScreenSp = yPos_ScreenSp * 10

        # Transform the coordinates from screen space to camera space.
        dX = float(currDeviceInfoDF['DeviceCameraToScreenXMm'])
        dY = float(currDeviceInfoDF['DeviceCameraToScreenYMm'])
        dW = float(currDeviceInfoDF['DeviceScreenWidthMm'])
        dH = float(currDeviceInfoDF['DeviceScreenHeightMm'])

        if orientation == 1:
            xPos_CamSp = xPos_ScreenSp - dX
            yPos_CamSp = -yPos_ScreenSp - dY
        elif orientation == 2:
            xPos_CamSp = xPos_ScreenSp + dX - dW
            yPos_CamSp = -yPos_ScreenSp + dY + dH
        elif orientation == 3:
            xPos_CamSp = xPos_ScreenSp + dY
            yPos_CamSp = -yPos_ScreenSp - dX + dW
        else:
            xPos_CamSp = xPos_ScreenSp - dY - dH
            yPos_CamSp = -yPos_ScreenSp + dX

        # Convert back to cm
        xPos_CamSp /= 10
        yPos_CamSp /= 10

        return [xPos_CamSp,yPos_CamSp]

    # =============================================================
    # Truncate predictions to screen
    def Truncate2Screen(self,xPos_ScreenSp,yPos_ScreenSp,orientation,currDeviceInfoDF):

        deviceScreenWidthMm = float(currDeviceInfoDF['DeviceScreenWidthMm'])
        deviceScreenHeightMm = float(currDeviceInfoDF['DeviceScreenHeightMm'])

        if orientation == 1 or orientation == 2:  # Camera is in portrait mode (home button either down or up)
            xPos_trunc = max(0, min(xPos_ScreenSp, deviceScreenWidthMm / 10))
            yPos_trunc = max(0, min(yPos_ScreenSp, deviceScreenHeightMm / 10))
        else:  # Camera is in landscape mode (home button either left or right)
            xPos_trunc = max(0, min(xPos_ScreenSp, deviceScreenHeightMm / 10))
            yPos_trunc = max(0, min(yPos_ScreenSp, deviceScreenWidthMm / 10))

        return [xPos_trunc,yPos_trunc]