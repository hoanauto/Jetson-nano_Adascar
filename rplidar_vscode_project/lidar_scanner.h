#ifndef LIDAR_SCANNER_H
#define LIDAR_SCANNER_H

#include <vector>
#include <string>
#include <memory>
#include "rplidar.h"

struct LidarPoint {
    float angle;      // degrees
    float distance;   // millimeters
    float quality;    // 0-255
    bool is_sync;     // sync bit
};

class LidarScanner {
private:
    rp::standalone::rplidar::RPlidarDriver* driver_;
    std::string port_;
    unsigned int baudrate_;
    bool is_connected_;
    
public:
    LidarScanner(const std::string& port = "/dev/ttyUSB1", unsigned int baudrate = 115200);
    ~LidarScanner();
    
    bool connect();
    void disconnect();
    bool isConnected() const { return is_connected_; }
    
    std::vector<LidarPoint> scan(int timeout_ms = 1000);
    std::vector<LidarPoint> scanSingle(int timeout_ms = 1000);
    
    std::string getDeviceInfo() const;
    std::string getHealthStatus() const;
    
    void startMotor();
    void stopMotor();
    
private:
    bool checkRPLIDARHealth();
};

#endif // LIDAR_SCANNER_H