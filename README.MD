# A Python Wrapper for iTracker
This repository contains the python wrapper for iTracker which we developed to carry out our benchmarking study "*Look me in the eye: Evaluating the phone-based eye tracking algorithm iTracker for monitoring gaze behaviour*" (Strobl et al, submitted). In this short tutorial we will outline how to use it to extract gaze estimations from a video taken with the front camera of a smart phone. For more information on iTracker, see the original paper [1], or the project webpage (http://gazecapture.csail.mit.edu/)

To get started, clone this repository to a convenient place on your computer, and follow the instructions below.

```
# Clone the repository
git clone https://github.com/ms234/iTrackerWrapper.git
```

## 1. The Docker Image  
![Docker Logo](images/dockerLogo.png)

iTracker was developed in Cafe, which has a number of inconvenient dependencies. In order to make it easier to get this code to run, and to aid reproducibility we generated a [Docker](https://www.docker.com/) image which has all the necessary dependencies installed. Docker in essence acts as a virtual machine and allows you to install libraries and software without mucking up your own. To install our image, install docker, and run the following bash command inside the `dockerImage` directory in this repository:

```bash
# Generate the docker image
sudo docker build -t gaze_estimator .
```

This will generate the image and call it "gaze_estimator". If this is the first time you're creating a Docker image this might take a bit, as it has to download all the required software (e.g. Ubuntu). Once installation has finished, you can run the image using:

```bash
# Start up the docker image
sudo docker run -it -p 2222:22 -p 8888:8888 -v /xxx/iTrackerWrapper:/workspace gaze_estimator bash
```

Replace "xxx" with the path to this repository on your computer. Then the above will start the image and will make the files inside this directory accessible from the docker image. The `-p 8888:8888` tag exposes port 8888 which allows you to access a jupyter server that runs on the docker container from your browser. The `-p 2222:22` tag allows you to ssh into the container using port 2222. I've found this useful for integrating the container with Pycharm, as it essentially handled the docker environment as a remote server. With integration of Docker into Pycharm this may no longer be necessary though. **If you're running a jupyter server already, this might cause you errors.** In this case change `-p 8888:8888` to `-p 8889:8888`. For more details on how to use Docker, see their webpage at https://www.docker.com/

Upon executing the above command you will launch the image, which for all intents and purposes of this work launches a virtual Ubuntu kernel on your machine. The console you're presented with is a standard Linux console. So, for example,

 ```bash
> root@72d574b7b024:/workspace/iTrackerWrapper# pwd
> /workspace/iTrackerWrapper
> root@72d574b7b024:/workspace/iTrackerWrapper# ls
> README.MD  data  dockerImage  extractFrames.py  images  segmentFrames.py
```
behave as expected. 

At this point, big thanks to the authors of the [cafe docker image](https://github.com/BVLC/caffe/tree/master/docker) and [dl-docker](https://github.com/floydhub/dl-docker), whose work served as a template for this image.

## 2. Pre-Processing the Data
Next, we have to extract the frames from the video. To do so, we use the `skvideo` module in Python. I've written a script to do this automatically for all files in the `data` directory. To run this, simply run

```bash
python extractFrames.py
``` 
The script is set up so to process all files in the `data` folder. So, **if you'd like to process some of your own videos, simply put them in the `data` folder.** I can guarantuee that it works for .mp4 files. Likely it will work for other video formats as well, but I haven't tested this. For demonstration purposes, I've included a video of myself from one of my own experiments.

**Note:** For some reason the `skvideo` struggles with the videos from certain cameras. In those cases the `width` and `height` axes are the wrong way around. To overcome this, I have copied the method from the source code, and modified it. **If your extracted frames look like garbage**, see the details in the comments in `extractFrames.py` on how to use this alternative `read()` function.

## 3. Segmenting the Images
iTracker uses 

To Do
- [] Add note that commands work on mac and linux. Not sure about windows
- [] Segmentation
- [] Run prediction both serial and in parallel
- [] Add r script to plot results
- [] Add docker reference to readme, and potentially picture as well
- [] Add picture of output at the beginning to make the readme look more visually appealing.

## References
[1] Krafka K, Khosla A, Kellnhofer P, Kannan H. Eye Tracking for Everyone. IEEE Conf Comput Vis Pattern Recognit. 2016;2176-2184.