#ifndef DATA_H
#define DATA_H

#define SHM_NAME "/lidar_shm"
#define MAX_POINTS 8192

struct Frame {
    int angle[MAX_POINTS];
    int distance[MAX_POINTS];
    unsigned int count;
};

extern Frame frame; // ð?nh ngh?a duy nh?t trong lidar.cpp

#endif // DATA_H
