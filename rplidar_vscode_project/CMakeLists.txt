cmake_minimum_required(VERSION 3.10)
project(rplidar_single_file)

set(CMAKE_CXX_STANDARD 11)

# T?m thý vi?n RPLIDAR
find_library(RPLIDAR_LIB rplidar
    PATHS /usr/local/lib
    REQUIRED
)

find_path(RPLIDAR_INCLUDE_DIR rplidar.h
    PATHS /usr/local/include
    REQUIRED
)

message(STATUS "Found RPLIDAR: ${RPLIDAR_LIB}")

# T?o executable t? 1 file duy nh?t
add_executable(lidar_single main.cpp)

# Include và link
target_include_directories(lidar_single PRIVATE ${RPLIDAR_INCLUDE_DIR})
target_link_libraries(lidar_single ${RPLIDAR_LIB} pthread)