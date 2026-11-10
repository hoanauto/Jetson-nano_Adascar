#!/bin/bash
echo "=== Building RPLIDAR Single File Scanner ==="

rm -f lidar_test

rm /home/nanob01/opencv_c_cuda/lidar_test

#Build
g++ working_stream.cpp \
    -std=c++11 \
    -o lidar_test \
    -I /home/nanob01/rplidar_installation/rplidar_sdk/sdk/include \
    -L /home/nanob01/rplidar_installation/rplidar_sdk/sdk/output/Linux/Release \
    -lrplidar_sdk \
    -lpthread \
    -lrt


cp lidar_test /home/nanob01/opencv_c_cuda

if [ $? -eq 0 ]; then
    echo "? Build successful!"
    echo "=== Running Lidar ==="
    sudo ./lidar_test

fi