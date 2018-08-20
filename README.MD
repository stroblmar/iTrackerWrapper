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

```
# Generate the docker image
sudo docker build -t gaze_estimator .
```

This will generate the image and call it "gaze_estimator". If this is the first time you're creating a Docker image this might take a bit, as it has to download all the required software (e.g. Ubuntu). Once installation has finished, you can run the image using:

```
# Start up the docker image
sudo docker run -it -p 2222:22 -p 8888:8888 -v /xxx/iTrackerWrapper:/workspace gaze_estimator bash
```

Replace "xxx" with the path to this repository on your computer. Then the above will start the image and will make the files inside this directory accessible from the docker image. The `-p 8888:8888` tag exposes port 8888 which allows you to access a jupyter server that runs on the docker container from your browser. The `-p 2222:22` tag allows you to ssh into the container using port 2222. I've found this useful for integrating the container with Pycharm, as it essentially handled the docker environment as a remote server. With integration of Docker into Pycharm this may no longer be necessary though  For more details on how to use Docker, see their webpage at https://www.docker.com/

At this point, big thanks to the authors of the [cafe docker image](https://github.com/BVLC/caffe/tree/master/docker) and [dl-docker]((https://github.com/floydhub/dl-docker)), whose work served as a template for this image.

## 2. Pre-Processing the Data

To Do
- [] Write a brief note on the read function issue when extracting frames
- [] Check that docker image installation etc works
- [] Add note that commands work on mac and linux. Not sure about windows
- [] Segmentation
- [] Run prediction both serial and in parallel
- [] Add r script to plot results
- [] Add docker reference to readme, and potentially picture as well
- [] Add picture of output at the beginning to make the readme look more visually appealing.

## References
[1] Krafka K, Khosla A, Kellnhofer P, Kannan H. Eye Tracking for Everyone. IEEE Conf Comput Vis Pattern Recognit. 2016;2176-2184.