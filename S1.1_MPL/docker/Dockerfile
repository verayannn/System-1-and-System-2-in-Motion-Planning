# ===================== Base Image =====================
FROM docker.1ms.run/ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:$PATH

# ===================== Domestic Mirror Source =====================
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list \
    && apt-get update


# ===================== Install Basic Tools =====================
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    git \
    wget \
    bzip2 \
    python3-catkin-tools \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    software-properties-common \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===================== Install Dependency Libraries =====================
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    gcovr \
    lcov \
    libvtk7-dev\
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# ===================== Install GCC-10 / G++-10 =====================
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && update-alternatives --set gcc /usr/bin/gcc-10 \
    && update-alternatives --set g++ /usr/bin/g++-10

RUN gcc --version && g++ --version

# ===================== Upgrade CMake =====================
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-aarch64.sh -O cmake.sh && \
    sh cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh
RUN cmake --version

# ===================== Install NLopt (from Source) =====================
WORKDIR /tmp
RUN git clone https://github.com/stevengj/nlopt.git && \
    cd nlopt && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/nlopt

# ===================== Install OSQP (from Source) =====================
WORKDIR /tmp
RUN git clone --recursive https://github.com/osqp/osqp.git && \
    cd osqp && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/osqp

# ===================== Install Catch2 (from Source) =====================
WORKDIR /tmp
RUN git clone https://github.com/catchorg/Catch2.git && \
    cd Catch2 && \
    cmake -Bbuild -H. -DBUILD_TESTING=OFF && \
    cmake --build build/ --target install && \
    cd / && rm -rf /tmp/Catch2

# ===================== Install Miniconda ARM64 =====================
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# ===================== Install ROS Python Dependencies in Conda =====================
RUN $CONDA_DIR/bin/pip install --no-cache-dir \
    catkin_pkg \
    rospkg \
    empy==3.3.4 \
    numpy

# ===================== Create Catkin Workspace =====================
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# ===================== Install catkin_simple and Build =====================
RUN git clone https://github.com/catkin/catkin_simple.git /catkin_ws/src/catkin_simple && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin init && catkin build"

# ===================== Add ROS Official/Tsinghua Mirror =====================
# Install curl, gnupg and related tools
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && rm -rf /var/lib/apt/lists/*

# Add ROS official/Tsinghua mirror source
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-keys F42ED6FBAB17C654 \
    && echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
    && apt-get update \
    && apt-get install -y \
        ros-noetic-desktop \
        ros-noetic-rviz \
        ros-noetic-gazebo-ros \
    && rm -rf /var/lib/apt/lists/*

# ===================== Set CMake and Library Paths =====================
ENV CMAKE_PREFIX_PATH=/usr/local
ENV NLopt_INCLUDE_DIR=/usr/local/include
ENV NLopt_LIBRARY=/usr/local/lib/libnlopt.so
ENV OSQP_INCLUDE_DIR=/usr/local/include
ENV OSQP_LIBRARY=/usr/local/lib/libosqp.so
ENV PCL_DIR=/usr/lib/aarch64-linux-gnu/cmake/pcl
ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$PCL_DIR

# ===================== Default to bash =====================
CMD ["bash"]
