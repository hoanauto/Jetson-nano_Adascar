#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "rplidar.h"
#include "data.h"

#ifndef _countof 
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0])) 
#endif

using namespace rp::standalone::rplidar;

Frame frame; // ð?nh ngh?a duy nh?t

bool keepRunning = true;

void ctrlc(int)
{
    keepRunning = false;
}

int main(int argc, const char * argv[]) {
    const char* opt_com_path = (argc > 1) ? argv[1] : "/dev/ttyUSB1";
    _u32 opt_com_baudrate = 115200;
    u_result op_result;

    std::vector<int> angles_LD;
    std::vector<int> distances_LD;
    int last_angle = 0;

    setbuf(stdout, NULL); // t?t buffer stdout

    // t?o driver
    RPlidarDriver* drv = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
    if (!drv) {
        fprintf(stderr, "Insufficient memory, exit\n");
        exit(-2);
    }

    if (IS_FAIL(drv->connect(opt_com_path, opt_com_baudrate))) {
        fprintf(stderr, "Cannot bind to serial port %s\n", opt_com_path);
        RPlidarDriver::DisposeDriver(drv);
        return -1;
    }

    drv->startMotor();
    drv->startScan(false, true, 0, nullptr);
    signal(SIGINT, ctrlc);

    fprintf(stderr, "LIDAR streaming started...\n");

    // t?o shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(Frame));
    Frame* shm_frame = (Frame*)mmap(nullptr, sizeof(Frame),
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    if (shm_frame == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    while (keepRunning) {
        rplidar_response_measurement_node_t nodes[MAX_POINTS];
        size_t count = _countof(nodes);

        op_result = drv->grabScanData(nodes, count);
        if (IS_OK(op_result)) {
            drv->ascendScanData(nodes, count);

            for (size_t pos = 0; pos < count; ++pos) {
                if (nodes[pos].distance_q2 != 0) {
                    float angle = (nodes[pos].angle_q6_checkbit >> RPLIDAR_RESP_MEASUREMENT_ANGLE_SHIFT) / 64.0f;
                    float distance = nodes[pos].distance_q2 / 1.0f;

                    int angle_int = static_cast<int>(angle * 100);
                    int distance_int = static_cast<int>(distance * 100);

                    angles_LD.push_back(angle_int);
                    distances_LD.push_back(distance_int);

                    // V?ng quét m?i
                    if (angle_int < last_angle) {
                        size_t n = angles_LD.size();
                        if (n > MAX_POINTS) n = MAX_POINTS;

                        for (size_t i = 0; i < n; ++i) {
                            frame.angle[i] = angles_LD[i];
                            frame.distance[i] = distances_LD[i];
                        }
                        frame.count = n;

                        // copy vào shared memory
                        if (shm_frame) *shm_frame = frame;

                        angles_LD.clear();
                        distances_LD.clear();
                    }

                    last_angle = angle_int;
                }
            }
        }
    }

    // d?ng lidar
    drv->stop();
    drv->stopMotor();
    drv->disconnect();
    RPlidarDriver::DisposeDriver(drv);

    fprintf(stderr, "LIDAR stream stopped.\n");
    return 0;
}
