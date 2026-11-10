#!/bin/bash
echo "=== Auto LIDAR System ==="

# Restart camera service (Jetson)
sudo systemctl restart nvargus-daemon

echo "Building test_cuda.cu..."

CUDA=/usr/local/cuda/bin/nvcc

$CUDA -o test_cuda test_cuda.cu \
  -I/usr/include/opencv4 \
  -lopencv_core \
  -lopencv_highgui \
  -lopencv_imgproc \
  -lopencv_videoio \
  -lopencv_cudaimgproc \
  -lopencv_cudafilters \
  -I/home/nanob01/rplidar_vscode_project/lidar_app \
  -lrt

if [ $? -ne 0 ]; then
    echo "? Build failed!"
    exit 1
fi

echo "? Build successful!"
echo ""
echo "Starting LIDAR system..."
echo "Press Ctrl+C to stop"
echo ""

# RUN BOTH WITH ROOT
sudo sh -c "./lidar_test 2>/dev/null | ./test_cuda"
