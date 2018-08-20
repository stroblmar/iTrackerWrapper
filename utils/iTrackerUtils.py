import os
import regex
# The caffe module needs to be on the Python path; we'll add it here explicitly.
import sys
import matplotlib as mpl
mpl.use('Agg')
caffePath = '/opt/caffe'  # path to caffe
sys.path.insert(0, caffePath + '/python')
import caffe
# =============================================================
# Function to convert the subject Id in a numeric format to the 5 digit format used by
# gazeCapture (e.g. go from 10 to '00010')
def NumTo5DigitCode(subjectId):
    subjectsIdStr = str(subjectId)
    nChar = len(subjectsIdStr)
    for i in range(5 - nChar): subjectsIdStr = '0'+subjectsIdStr
    return subjectsIdStr

# =============================================================
# Function to get the frame Ids for which we have reliable face
# and eye detection (i.e. all frames for which cropping worked)
def GetFrameIds(subjectId, dataPath="data/", appleFormat=True):
    # Get the file names
    # Check subjectId is in the right format (5-digit number)
    if type(subjectId) is not str: raise NameError("SubjectId in the wrong format. Must be a string.")
    if (len(subjectId) != 5): raise NameError("SubjectId in the wrong format. Must be of length 5")
    if appleFormat:
        faceDir = dataPath + "/" + str(subjectId) + "/appleFace"
        try:
            frameList = os.listdir(faceDir)
        except:
            print 'Can\'t find access the cropped images at: ' + faceDir + '. Maybe directory doesn\'t exist?'
    else:
        faceDir = dataPath + "/" + str(subjectId) + "/face"
        try:
            frameList = os.listdir(faceDir)
        except:
            print 'Can\'t find access the cropped images at: ' + faceDir + '. Maybe directory doesn\'t exist?'

    # Extract the frameIds as strings from the file names
    frameList = [regex.sub(".jpg", "", fName) for fName in frameList]
    return frameList


# =============================================================
# Function to load a frame
def LoadFrame(subjectId, frameId, dataPath="data", appleFormat=True):
    # Set up the paths
    subjectDir = dataPath + "/" + subjectId
    imgName = frameId + '.jpg'
    if appleFormat:
        leftEyeImgPath = subjectDir + '/appleLeftEye/' + imgName
        rightEyeImgPath = subjectDir + '/appleRightEye/' + imgName
        faceImgPath = subjectDir + '/appleFace/' + imgName
        faceGridImgPath = subjectDir + '/appleFaceGrid/' + imgName
    else:
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
    return {'leftEyeImg': leftEyeImg, 'rightEyeImg': rightEyeImg, 'faceImg': faceImg, 'faceGridImg': faceGridImg}