# ================================================
# Script to extract frames from the videos inside the "data" directory.
# NOTE: For some phones I found that the skvideo.read() method imports
# the videos with height and width inverted. For these cases, I used
# the modified version of the read function added below in the script.
# So, if the frames coming out of the extraction process look like garbage
# try uncommenting the lines marked with "<Alternative read() method>".
# =============================================================
# Configure Script
# Configurations for file system and image segmentation
dataPath = "/workspace/data" # Path to where the videos are housed
imgExtension = ".jpg"
extractRate = 1 # Extract every nth frame from the video

# ================================================
# Libraries
import os
import skvideo.io
import scipy.misc
import numpy
# ================================================
# Function adapted from the declaration of the videoCapture class of skivideo. Switched height and
# width in the reshape command, so that it works with the videos from Roche.
def read(vidCapObj):
  retval = True

  nbytes = vidCapObj.width * vidCapObj.height * vidCapObj.depth

  while len(vidCapObj.buf) < nbytes:

    # Could poll here, but return code never seems to be set before we fail at reading anyway
    # vidCapObj.proc.poll()

    if vidCapObj.proc.returncode != None:
      if vidCapObj.proc.returncode < 0:
        raise ValueError(
          "Command exited with return code %d" % (vidCapObj.proc.returncode))  # TODO subprocess.CalledProcessError?
      else:
        return False, None

    buf = vidCapObj.proc.stdout.read(nbytes - len(vidCapObj.buf))
    # print "Read %d" % (len(buf))

    # Reading no data seems to be a reliable end-of-file indicator; return code is not.
    if len(buf) == 0:
      break

    vidCapObj.buf += buf

  if len(vidCapObj.buf) < nbytes:
    # We didn't get any data, assume end-of-file
    if len(vidCapObj.buf) == 0:
      return False, None
    # We got some data but not enough, this is an error
    else:
      raise ValueError("Not enough data at end of file, expected %d bytes, read %d" % (nbytes, len(vidCapObj.buf)))

  image = numpy.fromstring(vidCapObj.buf[:nbytes], dtype=numpy.uint8).reshape((vidCapObj.width, vidCapObj.height, vidCapObj.depth))

  # If there is data left over, move it to beginning of buffer for next frame
  if len(vidCapObj.buf) > nbytes:
    vidCapObj.buf = vidCapObj.buf[nbytes:]  # TODO this is a relatively slow operation, optimize
  # Otherwise just forget the buffer
  else:
    vidCapObj.buf = ""

  return retval, image

# =============================================================
# Main Part of the Program
# =============================================================
# Assemble list of video files in data directory
videoList = [fName for fName in os.listdir(dataPath) if not os.path.isdir(dataPath+"/"+fName)]
try:
    videoList.remove('.DS_Store') # Annoying meta-data files which will cause problems if you're working on a mac
except:
    pass
print(videoList)

#  Extract frames from videos
for videoNameStr in videoList:
    print("------------------------------------------------------------")
    print("Extract frames from %s"%videoNameStr)
    vidcap = skvideo.io.VideoCapture(dataPath + "/" + videoNameStr)
    outputPath = dataPath + "/" + videoNameStr.split(".")[0]
    # Generate folders to hold frames
    try:
        os.mkdir(outputPath)
    except:
        pass
    try:
        os.mkdir(outputPath + "/frames")
    except:
        pass
    # Extract frames
    nFrames = 0
    success, frame = vidcap.read()
    # success, frame = read(vidcap) # Uncomment to use <Alternative read() method>
    while success:
        # print('Read a new frame: ', success)
        if (nFrames % extractRate == 0):
            scipy.misc.toimage(frame, cmin=0.0, cmax=255).save(
                outputPath + "/frames/img%d.jpg" % nFrames)
        nFrames += 1
        # Read next frame
        success, frame = vidcap.read()
        # success, frame = read(vidcap) # Uncomment to use <Alternative read() method>

