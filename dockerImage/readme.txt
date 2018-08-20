# ====================================
# Command for using the docker image
# ====================================
# Building the image
# -------------------
# On Ubuntu execute in this folder:
sudo docker build -t gaze_estimator .

# Running the image
# -------------------
# To link the local code directory and data directory with docker
sudo docker run -it -p 2222:22 -p 8888:8888 -v /local/jesu2844/Desktop/gaze_tracking_Project/code/:/workspace -v /data/:/data gaze_estimator bash

# Same but to link my dropbox folder
sudo docker run -it -p 2222:22 -p 8888:8888 -v /local/jesu2844/Dropbox/gaze_tracking_Project/code/:/workspace -v /data/:/data gaze_estimator bash

# For my laptop
docker run -it -p 2222:22 -p 8888:8888 -v /Users/Maxi/Dropbox/gaze_tracking_Project/code/:/workspace gaze_estimator bash


# SSH connection into the container
# -------------------
ssh root@localhost -p 2222


# Using Jupyter Notebooks
# -------------------
/usr/local/bin/jupyter notebook
http://0.0.0.0:8888/
