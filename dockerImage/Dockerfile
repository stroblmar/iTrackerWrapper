FROM ubuntu:16.04
LABEL maintainer Maximilian A R Strobl
# Build by using a combination of the cafe docker image (https://github.com/BVLC/caffe/tree/master/docker) and dl-docker (https://github.com/floydhub/dl-docker)

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# ================================================
# Install Caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# ================================================
# Install and set up jupyter
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
		ipykernel \
		jupyter \
		&& \
	python -m ipykernel.kernelspec

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# ================================================
# Install ssh so can ssh into docker container via pycharm. Instructions taken from:
# https://docs.docker.com/engine/examples/running_ssh_service/#build-an-eg_sshd-image
RUN apt-get update;apt-get install -y openjdk-8-jdk-headless wget openssh-server tar vim

# Set up ssh
RUN mkdir /var/run/sshd
# Set password to screencast
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# ================================================
# Install python-tk so to run matplotlib
RUN apt-get update;apt-get install -y python-tk

# ================================================
# Install unzip to allow unzipping
RUN apt-get update;apt-get install -y unzip

# ================================================
# Install R
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | tee -a /etc/apt/sources.list
RUN gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
RUN gpg -a --export E084DAB9 | apt-key add -
RUN apt-get update;apt-get install -y r-base r-base-dev
COPY install-packages.R /root/
RUN R CMD BATCH /root/install-packages.R

# ================================================
# Install other useful Python packages using pip
RUN pip --no-cache-dir install \
        regex \
        opencv-python \
        lmdb

# ================================================
# Install dependencies needed for video processing
RUN apt-get update;apt-get install -y libav-tools
RUN pip --no-cache-dir install \
        scikit-video

# ================================================
# Expose Ports for Ipython (8888) and ssh (22 8088 50070)
EXPOSE 8888 22 8088 50070
WORKDIR /workspace
CMD ["/bin/bash"]
#CMD ["sh", "-c" ,"/usr/local/bin/jupyter notebook; /bin/bash"]
