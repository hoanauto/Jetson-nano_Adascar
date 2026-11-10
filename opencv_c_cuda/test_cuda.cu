#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp> 
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include "../rplidar_vscode_project/data.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <algorithm>

#define Q 16
#define SCALE (1 << Q)

using namespace std;
using namespace cv;

Frame* shm_frame = nullptr;
unsigned int last_count = 0;

// ---------- openSerial (giữ nguyên) ----------
int openSerial(const char* port, int baud = B115200) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        perror("open");
        return -1;
    }
    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        close(fd);
        return -1;
    }
    cfsetospeed(&tty, baud);
    cfsetispeed(&tty, baud);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return -1;
    }
    return fd;
}

// ---------- Globals for serial queue ----------
std::queue<std::string> q;
std::mutex mtx;
std::atomic<bool> running(true);

// ---------- Serial thread with reconnect ----------
void serialThread(std::atomic<int>& afd, const char* port, int baud = B115200) {
    while (running.load()) {
        int cur_fd = afd.load();
        if (cur_fd < 0) {
            // Thử kết nối lại
            std::cout << "[Serial] 🔄 Thử kết nối lại: " << port << " ...\n";
            int new_fd = openSerial(port, baud);
            if (new_fd >= 0) {
                afd.store(new_fd);
                std::cout << "[Serial] ✅ Đã kết nối: " << port << "\n";
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                continue;
            }
        }

        // Lấy message từ hàng đợi
        std::string msg;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!q.empty()) {
                msg = q.front();
                q.pop();
            }
        }

        if (!msg.empty()) {
            int fd = afd.load();
            if (fd >= 0) {
                ssize_t n = write(fd, msg.c_str(), msg.size());
                if (n < 0) {
                    perror("[Serial] write");
                    std::cerr << "[Serial] ❌ Ghi thất bại — đóng fd và sẽ reconnect\n";
                    close(fd);
                    afd.store(-1);
                    // optional: push message back so không mất dữ liệu (tuỳ bạn)
                    // std::lock_guard<std::mutex> lock(mtx);
                    // q.push(msg);
                }
            } else {
                // nếu fd âm, đẩy lại message để gửi sau khi reconnect
                std::lock_guard<std::mutex> lock(mtx);
                q.push(msg);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        } else {
            // không có message — nghỉ 1 chút
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // khi kết thúc thread, đóng fd nếu cần
    int fd_end = afd.load();
    if (fd_end >= 0) close(fd_end);
}

// ---------------- GPU / vision kernels & helpers (giữ nguyên nhưng cần đảm bảo forward decls) ----------------
__global__ void drawlinecenter(uchar* frame, int height, int width, int x_line, size_t step) {
    uint y = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= 200 && y < 460 && x_line >= 0 && x_line < width) {
        for (int8_t i = -1; i <= 1; i++) {
            int x = x_line + i;
            if (x >= 0 && x < width) {
                uchar* pixel = frame + y * step + x * 3;
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            }
        }
    }
    else if (y >= 460 && y < height && x_line >= 0 && x_line < width) {
        for (int8_t i = -1; i <= 1; i++) {
            int x = x_line + i;
            if (x >= 0 && x < width) {
                uchar* pixel = frame + y * step + x * 3;
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            }
        }
    }
}

cuda::GpuMat drawLineCenter(
    const cv::cuda::GpuMat &input, 
    int& x_cen
){
    if (x_cen < 0 || x_cen > 640) return input;
    cv::cuda::GpuMat output;
    input.copyTo(output);
    uint thread = 256;
    uint block = (480 + thread - 1) / thread;
    drawlinecenter<<<block, thread>>>(output.ptr<uchar>(), 480, 640, x_cen, output.step); //340
    return output;
}

struct Line { int x0, y0, x1, y1; };

__global__ void scanLane1D(
    const uchar* mask,
    size_t step,
    int width,
    int height,
    int delta_ob, 
    int x_cen,
    int* x1_out,
    int* x2_out
){
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    const uchar* row = mask + y * step;

    int x1 = -1, x2 = -1;
    int found = 0;

    if (delta_ob > 0) {
        if (delta_ob <= 10) {
            int xs = max(width-1, x_cen + 50);
            int end = x_cen + 50;
            for (int i =  xs; i > end; i--) {
                if (row[i]) {
                    if (!found) { x1 = i; found = 1; }
                    else { x2 = i; break; }
                }
            }
        }
        else if (delta_ob >= 12){
            for (int i = x_cen - 100; i < width; i++) {
                if (row[i]) {
                    if (!found) { x1 = i; found = 1; }
                    else { x2 = i; break; }
                }
            }
        }
        
    }
    else if (delta_ob < 0) {

        if (delta_ob >= -10){
            int xs = min(0, x_cen - 50);
        int end = x_cen - 50;
        for (int i = 0; i < end; i++) {
            if (row[i]) {
                if (!found) { x1 = i; found = 1; }
                else { x2 = i; break; }
            }
        }
        }
        

        else if (delta_ob <= -12){
            for (int i = x_cen + 100; i > 0; i--) {
                if (row[i]) {
                    if (!found) { x1 = i; found = 1; }
                    else { x2 = i; break; }
                }
            }
        }
    }

    x1_out[y] = x1;
    x2_out[y] = x2;
}


void scanLaneByDirection(
    cv::cuda::GpuMat& gpuMask,
    int delta_ob, int x_cen,  // +1 ho?c -1
    std::vector<uint16_t>& x_center_up,std::vector<uint16_t>& y_center_up,
    std::vector<uint16_t>& x_center_down,std::vector<uint16_t>& y_center_down,
    uint16_t& len_up, uint16_t& len_down

){

    std::vector<int> x_center;
    std::vector<int> y_center;

    int ymin_found = -1;
    int ymax_found = -1;
    int x_ymin = 0, x_ymax = 0;

    int D_max = 10;
    int d_max = 0;
    int x_D_max = 0;
    int y_D_max = 0;
    int vtr = 0; // turning index
    int vtr_down = 0;

    const int D_const = 7;
    int width  = gpuMask.cols;
    int height = gpuMask.rows;

    int *d_x1, *d_x2;
    cudaMalloc(&d_x1, height * sizeof(int));
    cudaMalloc(&d_x2, height * sizeof(int));

    int threads = 256;
    int blocks  = (height + threads - 1) / threads;

    scanLane1D<<<blocks, threads>>>(
        gpuMask.ptr<uchar>(),
        gpuMask.step,
        width,
        height,
        delta_ob,
        x_cen,
        d_x1,
        d_x2
    );

    std::vector<int> x1(height), x2(height);
    cudaMemcpy(x1.data(), d_x1, height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2.data(), d_x2, height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x1);
    cudaFree(d_x2);

    x_center.clear();
    y_center.clear();
    for (int y = 0; y < height; y++) {
        int w = abs(x2[y] - x1[y]);
        if (w >= 10 && w <= 20) {
            x_center.push_back((x1[y] + x2[y]) / 2);
            y_center.push_back(y);
        }
    }
    uint16_t N = y_center.size();

    if (N == 0) {
            len_up = 0;
            len_down = 0;
            return;
        }

    ymin_found = y_center[0];
    x_ymin = x_center[0];
    ymax_found = y_center[y_center.size() - 1];
    x_ymax = x_center[x_center.size() - 1];
    //std::cout<<"x_ymax:"<<x_ymax<<endl;


    float a = ymin_found - ymax_found;
    float b = x_ymax - x_ymin;
    float c = (x_ymin * ymax_found) - (x_ymax *ymin_found);

    // Distance D
    
    for (int i = 0; i < x_center.size(); i++){
        int x_yellow = x_center[i];
        int y_yellow = y_center[i];
        int D = abs((int)((a * x_yellow + b * y_yellow + c) / sqrt(a*a + b*b)));

        if (D > d_max) {
            d_max = D;
            x_D_max = x_yellow;
            y_D_max = y_yellow;
            vtr = i; // index turning
        } 
    }
    if (d_max < D_max) {
        vtr = -1;
    }

    //====================================================
    //  Split UP & DOWN from vtr
    //====================================================

    

    
    if (vtr >= 0) {
        if (vtr >= N) vtr = N-1;  
        len_up = vtr;
        len_down = N - vtr;
        vtr_down = vtr;
        
    }
    else if (vtr < 0) {
        len_up = N;
        len_down = N;
        vtr_down = 0;
    }
    /*std::cout<<"vtr: "<<vtr<<endl;
        std::cout<<"len_up: "<<len_up<<endl;
        std::cout<<"len_down: "<<len_down<<endl;
        std::cout<<"N: "<<N<<endl;*/
    
        x_center_up.clear();
        y_center_up.clear();
        x_center_down.clear();
        y_center_down.clear();

        // UP (0 → vtr-1)
        for (int i = 0; i < len_up; i++) {
            x_center_up.push_back(x_center[i]);
            y_center_up.push_back(y_center[i]);
            //std::cout<<"yellow_up_"<<i<<": "<<yellow_y[i]<<endl;
        }

        // DOWN (vtr → end)
        for (int i = 0; i < len_down; i++) {
            x_center_down.push_back(x_center[vtr_down + i]);
            y_center_down.push_back(y_center[vtr_down + i]);
            //std::cout<<"yellow_down_"<<i<<": "<<yellow_y[vtr + i]<<endl;
        }
        //std::cout<<"yellow_up_"<<len_up - 1<<": "<<yellow_y_up_host[len_up-1]<<endl;
        //std::cout<<"yellow_down_"<<0<<": "<<yellow_y_down_host[len_down - 1]<
}




__global__ void detectLaneKernel2D(
    const uchar* mask, size_t maskStep,
    int width, int height,
    int* leftOuter, int* leftInner,
    int* rightInner, int* rightOuter
){
    // m?i block x? l? 1 d?ng y
    int y = blockIdx.x;
    if (y >= height) return;

    const int x_center = 300;   // ho?c width / 2
    const uchar* row = mask + y * maskStep;

    // ===== BIÊN TRÁI =====
    int lInner = -1;
    int lOuter = -1;
    bool inLane = false;

   for (int x = x_center; x >= 0; --x) {
    if (row[x] > 0) {
        lInner = x;
        lOuter = max(0, x - 15);
        break;
    }
}

    // ===== BIÊN PH?I =====
    int rInner = -1;
    int rOuter = -1;
    inLane = false;

    for (int x = x_center; x < width; ++x) {
    if (row[x] > 0) {
        rInner = x;
        rOuter = min(width - 1, x + 15);
        break;
    }
}

    // reset m?c ð?nh
    leftOuter[y]  = -1;
    leftInner[y]  = -1;
    rightInner[y] = -1;
    rightOuter[y] = -1;

    // ===== KI?M TRA LANE H?P L? =====
    if (lInner >= 0 && rInner >= 0) {

        leftInner[y]  = lInner;
            leftOuter[y]  = lOuter;
            rightInner[y] = rInner;
            rightOuter[y] = rOuter;
    }
}




__global__ void drawLinesKernel2D(
    uchar* frame, int width, int height, size_t step,
    Line* lines, uchar3* colors, int nLines, int thickness
){
    int lineId = blockIdx.x;
    int tid    = threadIdx.x;

    if (lineId >= nLines) return;

    Line L = lines[lineId];
    uchar3 color = colors[lineId];
    int dx = abs(L.x1 - L.x0);
    int dy = abs(L.y1 - L.y0);
    int N  = max(dx, dy) + 1;

    if (tid < N) {
        float t = (float)tid / (N - 1);
        int x = L.x0 + (int)roundf(t * (L.x1 - L.x0));
        int y = L.y0 + (int)roundf(t * (L.y1 - L.y0));
        
        for (int dy2 = -thickness; dy2 <= thickness; ++dy2) {
            for (int dx2 = -thickness; dx2 <= thickness; ++dx2) {
                int xx = x + dx2, yy = y + dy2;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    uchar* row = frame + yy * step;
                    row[3*xx + 0] = color.x;
                    row[3*xx + 1] = color.y;
                    row[3*xx + 2] = color.z;
                }
            }
        }
    }
}

__device__ int x_turning_angle, y_turning_angle;

inline void DrawTwoLaneCenterGPU(
    cv::cuda::GpuMat& frameBGR,
    const cv::cuda::GpuMat& mask,
    std::vector<uint16_t>& yellow_x_up_host,
    std::vector<uint16_t>& yellow_y_up_host,
    std::vector<uint16_t>& yellow_x_down_host,
    std::vector<uint16_t>& yellow_y_down_host,
    uint16_t& len_up,
    uint16_t& len_down
){
    int rows = mask.rows;
    int cols = mask.cols;


    int ymin_found = -1;
    int ymax_found = -1;
    int x_ymin = 0, x_ymax = 0;

    int D_max = 10;
    int d_max = 0;
    int x_D_max = 0;
    int y_D_max = 0;
    int vtr = 0; // turning index
    int vtr_down = 0;

    const int D_const = 7;

    //--------------- GPU malloc ---------------
    int *d_leftOuter, *d_leftInner, *d_rightInner, *d_rightOuter;
    cudaMalloc(&d_leftOuter, rows * sizeof(int));
    cudaMalloc(&d_leftInner, rows * sizeof(int));
    cudaMalloc(&d_rightInner, rows * sizeof(int));
    cudaMalloc(&d_rightOuter, rows * sizeof(int));

    cudaMemset(d_leftOuter,  0x7F, rows * sizeof(int));
    cudaMemset(d_rightOuter, 0,    rows * sizeof(int));
    cudaMemset(d_leftInner,  0,    rows * sizeof(int));
    cudaMemset(d_rightInner, 0x7F, rows * sizeof(int));


detectLaneKernel2D<<<rows, 1>>>(
    mask.ptr<uchar>(), mask.step,
    cols, rows,
    d_leftOuter, d_leftInner,
    d_rightInner, d_rightOuter
);
cudaDeviceSynchronize();


    //--------------- Copy back ---------------
    std::vector<int> h_leftOuter(rows), h_leftInner(rows),
                     h_rightInner(rows), h_rightOuter(rows);

    cudaMemcpy(h_leftOuter.data(),  d_leftOuter,  rows*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leftInner.data(),  d_leftInner,  rows*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rightInner.data(), d_rightInner, rows*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rightOuter.data(), d_rightOuter, rows*sizeof(int), cudaMemcpyDeviceToHost);

    //--------------- Yellow line arrays ---------------
    std::vector<int> yellow_x;
    std::vector<int> yellow_y;

    std::vector<Line> h_lines;
    std::vector<uchar3> h_colors;

    //--------------- Main scan loop ---------------
    for (int y = 0; y < rows-1; y++) {

        bool leftValid =
            (h_leftOuter[y] < cols && h_leftInner[y] > 0 &&
             h_leftOuter[y+1] < cols && h_leftInner[y+1] > 0);

        bool rightValid =
            (h_rightInner[y] < cols && h_rightOuter[y] > 0 &&
             h_rightInner[y+1] < cols && h_rightOuter[y+1] > 0);

        int xc1 = 0, xc1b = 0, xc2 = 0, xc2b = 0;

        if (leftValid) {
            xc1  = (h_leftOuter[y]  + h_leftInner[y])  / 2;
            xc1b = (h_leftOuter[y+1]+ h_leftInner[y+1]) / 2;

            h_lines.push_back({xc1, y, xc1b, y+1});
            h_colors.push_back(make_uchar3(0,255,0));
        }

        if (rightValid) {
            xc2  = (h_rightInner[y]  + h_rightOuter[y])  / 2;
            xc2b = (h_rightInner[y+1]+ h_rightOuter[y+1]) / 2;

            h_lines.push_back({xc2, y, xc2b, y+1});
            h_colors.push_back(make_uchar3(0,255,0));
        }

        if (leftValid && rightValid) {
            int xm  = (xc1 + xc2) / 2;
            int xmb = (xc1b + xc2b) / 2;

            int dx = abs(xmb - xm);
            int dc1 = abs(xm - xc1);
            int dc2 = abs(xc2 - xm);

            if (dx < 3 && dc1 > 20 && dc2 > 20) {

                if (ymin_found == -1) {
                    ymin_found = y;
                    x_ymin = xm;
                }

                ymax_found = y;
                x_ymax = xm;

                yellow_x.push_back(xm);
                yellow_y.push_back(y);

                h_lines.push_back({xm, y, xmb, y+1});
                h_colors.push_back(make_uchar3(0,255,255));     
            }
        }
    }

    //------------- turning detection -------------
    float a = ymin_found - ymax_found;
    float b = x_ymax - x_ymin;
    float c = (x_ymin * ymax_found) - (x_ymax *ymin_found);

    // Distance D
    
    for (int i = 0; i < yellow_x.size(); i++){
        int x_yellow = yellow_x[i];
        int y_yellow = yellow_y[i];
        int D = abs((int)((a * x_yellow + b * y_yellow + c) / sqrt(a*a + b*b)));

        if (D > d_max) {
            d_max = D;
            x_D_max = x_yellow;
            y_D_max = y_yellow;
            vtr = i; // index turning
        } 
    }
    if (d_max < D_max) {
        vtr = -1;
    }

    //====================================================
    //  Split UP & DOWN from vtr
    //====================================================

    uint16_t N = yellow_x.size();

    if (N == 0) {
            len_up = 0;
            len_down = 0;
            return;
        }
    if (vtr >= 0) {
        if (vtr >= N) vtr = N-1;  
        len_up = vtr;
        len_down = N - vtr;
        vtr_down = vtr;
        
    }
    else if (vtr < 0) {
        len_up = N;
        len_down = N;
        vtr_down = 0;
    }
    /*std::cout<<"vtr: "<<vtr<<endl;
        std::cout<<"len_up: "<<len_up<<endl;
        std::cout<<"len_down: "<<len_down<<endl;
        std::cout<<"N: "<<N<<endl;*/
    
        yellow_x_up_host.clear();
        yellow_y_up_host.clear();
        yellow_x_down_host.clear();
        yellow_y_down_host.clear();

        // UP (0 → vtr-1)
        for (int i = 0; i < len_up; i++) {
            yellow_x_up_host.push_back(yellow_x[i]);
            yellow_y_up_host.push_back(yellow_y[i]);
            //std::cout<<"yellow_up_"<<i<<": "<<yellow_y[i]<<endl;
        }

        // DOWN (vtr → end)
        for (int i = 0; i < len_down; i++) {
            yellow_x_down_host.push_back(yellow_x[vtr_down + i]);
            yellow_y_down_host.push_back(yellow_y[vtr_down + i]);
            //std::cout<<"yellow_down_"<<i<<": "<<yellow_y[vtr + i]<<endl;
        }
        //std::cout<<"yellow_up_"<<len_up - 1<<": "<<yellow_y_up_host[len_up-1]<<endl;
        //std::cout<<"yellow_down_"<<0<<": "<<yellow_y_down_host[len_down - 1]<<endl;
    

    //====================================================
    //   DRAW LINES ON GPU
    //====================================================
    if (!h_lines.empty()) {
        size_t n = h_lines.size();

        Line* d_lines = nullptr;
        uchar3* d_colors = nullptr;

        cudaMalloc(&d_lines, n * sizeof(Line));
        cudaMalloc(&d_colors, n * sizeof(uchar3));

        cudaMemcpy(d_lines,  h_lines.data(),  n*sizeof(Line), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, h_colors.data(), n*sizeof(uchar3), cudaMemcpyHostToDevice);

        int threads = 256;
drawLinesKernel2D<<<n, threads>>>(
    frameBGR.ptr<uchar>(), cols, rows, frameBGR.step,
    d_lines, d_colors, n, 2
);

        cudaFree(d_lines);
        cudaFree(d_colors);
    }

    //----------- free GPU memory -----------
    cudaFree(d_leftOuter);
    cudaFree(d_leftInner);
    cudaFree(d_rightInner);
    cudaFree(d_rightOuter);
}

__global__ void FindLine(
    const uint16_t* x, const uint16_t* y, int N,
    float* sum_x, float* sum_y, float* sum_xy,float* sum_xx, float* sum_yy
){
    __shared__ float local_x[256];
    __shared__ float local_y[256];
    __shared__ float local_xy[256];
    __shared__ float local_xx[256];
    __shared__ float local_yy[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    float x_value = 0.0f;
    float y_value = 0.0f;
    if (idx < N) {
        x_value = float(x[idx]);
        y_value = float(y[idx]);
    }

    local_x[t] = x_value;
    local_y[t] = y_value;
    local_xy[t] = x_value * y_value;
    local_xx[t] = x_value * x_value;
    local_yy[t] = y_value * y_value;

    __syncthreads();

    for (int stride =  blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            local_x[t] += local_x[t + stride];
            local_y[t] += local_y[t + stride];
            local_xy[t] += local_xy[t + stride];
            local_xx[t] += local_xx[t + stride];
            local_yy[t] += local_yy[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(sum_x, local_x[0]);
        atomicAdd(sum_y, local_y[0]);
        atomicAdd(sum_xy, local_xy[0]);
        atomicAdd(sum_xx, local_xx[0]);
        atomicAdd(sum_yy, local_yy[0]);
    }
}

void Linear(
    const vector<uint16_t>& hx, vector<uint16_t>& hy,
    float &a, float &b, float &c
  
){
    int N = hx.size();

    uint16_t *dx, *dy;
    cudaMalloc(&dx, N * sizeof(uint16_t));
    cudaMalloc(&dy, N * sizeof(uint16_t));

    cudaMemcpy(dx, hx.data(), N*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), N*sizeof(uint16_t), cudaMemcpyHostToDevice);


    float *dsum_x, *dsum_y, *dsum_xy, *dsum_xx, *dsum_yy;
    cudaMalloc(&dsum_x, sizeof(float));
    cudaMalloc(&dsum_y, sizeof(float));
    cudaMalloc(&dsum_xy, sizeof(float));
    cudaMalloc(&dsum_xx, sizeof(float));
    cudaMalloc(&dsum_yy, sizeof(float));

    cudaMemset(dsum_x, 0, sizeof(float));
    cudaMemset(dsum_y, 0, sizeof(float));
    cudaMemset(dsum_xy, 0, sizeof(float));
    cudaMemset(dsum_xx, 0, sizeof(float));
    cudaMemset(dsum_yy, 0, sizeof(float));

    int block = 256;
    int grid = (N + block - 1) / block;

    FindLine<<<grid, block>>> (
        dx, dy, N,
        dsum_x, dsum_y, dsum_xy, dsum_xx, dsum_yy
    );
    cudaDeviceSynchronize();

    float sum_x, sum_y, sum_xy, sum_xx, sum_yy;
    cudaMemcpy(&sum_x, dsum_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_y, dsum_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_xy, dsum_xy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_xx, dsum_xx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_yy, dsum_yy, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dsum_x);
    cudaFree(dsum_y);
    cudaFree(dsum_xy);
    cudaFree(dsum_xx);
    cudaFree(dsum_yy);

    /*std::cout<<"sum_x: "<<sum_x<<endl;
    std::cout<<"sum_y: "<<sum_y<<endl;
    std::cout<<"sum_xy: "<<sum_xy<<endl;
    std::cout<<"sum_xx: "<<sum_xx<<endl;
    std::cout<<"sum_yy: "<<sum_xx<<endl;*/
    

    float mx = sum_x / N;
    float my = sum_y / N;

    float sxx = sum_xx - N * mx*mx;
    float sxy = sum_xy - N * mx*my;
    float syy = sum_yy - N * my*my;

    float lamda_1 = (sxx + syy + sqrt((sxx - syy) * (sxx - syy) + 4 * sxy * sxy)) / 2; 
    float lamda_2 = (sxx + syy - sqrt((sxx - syy) * (sxx - syy) + 4 * sxy * sxy)) / 2; 
    float lamda_min;
    if (lamda_1 <= lamda_2) {
        lamda_min = lamda_1;
    }
    else lamda_min = lamda_2;

    float nx = lamda_min - syy;
    float ny = sxy;

    float A = nx;
    float B = ny;
    float C = -(A*mx + B*my);
    float x_f = - C / A;
    a = A;
    b = B;
    c = C;
    
    /*std::cout<<"A: "<<A<<endl;
    std::cout<<"B: "<<B<<endl;
    std::cout<<"C: "<<C<<endl;
    std::cout<<"x_f: "<<x_f<<endl;
    std::cout<<endl;*/
}

__host__ void intersection (
    int width, int height,
    float A_up, float B_up, float C_up,
    float A_down, float B_down, float C_down, 
    float &y_inte, float &x_inte, int &inte
) {
    float D = (A_up * B_down) - (A_down * B_up);

    if (D == 0){
        inte = 0;
    }
    else {
        x_inte = ((B_up * C_down) - (B_down * C_up)) / D;
        y_inte = ((C_up * A_down) - (C_down * A_up)) / D;
        if (x_inte <= 0 || x_inte >= width || y_inte <=0 || y_inte >= height) {
            inte = 0;
        }
        else {inte = 1;}
    }
}

__global__ void drawLinear(
    uchar* frame, int width, int height, size_t step, 
    float A, float B, float C,
    int x1, int x2,
    int y1, int y2, 
    uchar3 color
){
    // Kiểm tra gần đứng: |B| rất nhỏ so với |A|
    if (fabsf(B) < fabsf(A) * 0.1f){
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= y1 && y <= y2) {
            // x(y) = -(B*y + C) / A
            float xf = -(B*y + C) / A;
            int x = (int)(xf + 0.5f);
            for (int8_t i = -1; i <= 1; i++) {
                int x_line = x + i;
                if (x >= 0 && x < width) {
                    uchar* px = frame + y * step + x_line * 3;
                    px[0] = color.x;
                    px[1] = color.y; 
                    px[2] = color.z;
                }
            }
        }
    }
        
    // Nếu không phải đứng: chọn hướng theo slope (chính xác!)
    float slope = -A / B;

    if (fabsf(slope) > 1.0f) {
        // dốc nhiều → scan theo y
        int y = blockIdx.x * blockDim.x + threadIdx.x;

        if (y < min(y1, y2) || y > max(y1, y2) || y >= height) return;

        float xf = -(B * y + C) / A;
        int x = (int)(xf + 0.5f);

        for (int8_t i = -1; i <= 1; i++) {
                int x_line = x + i;
                if (x >= 0 && x < width) {
                    uchar* px = frame + y * step + x_line * 3;
                    px[0] = color.x;
                    px[1] = color.y; 
                    px[2] = color.z;
                }
            }
    }
    else {
        // ít dốc → scan theo x
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x < min(x1, x2) || x > max(x1, x2) || x >= width) return;

        float yf = -(A * x + C) / B;
        int y = (int)(yf + 0.5f);

        for (int8_t i = -1; i <= 1; i++) {
                int x_line = x + i;
                if (x >= 0 && x < width) {
                    uchar* px = frame + y * step + x_line * 3;
                    px[0] = color.x;
                    px[1] = color.y; 
                    px[2] = color.z;
                }
            }
    }
}

void drawLinearOnGpu (
    cv::cuda::GpuMat& frame,
    float A, float B, float C,
    int x1, int x2,
    int y1, int y2,
    uchar3 color
){
    int width =  frame.cols;
    int height = frame.rows;
    size_t step = frame.step;

    uchar* d_ptr = frame.ptr<uchar>();

    int threads = 256;
    uint blocks = (480 + threads - 1) / threads;

    drawLinear<<<blocks, threads>>>(
        d_ptr, width, height, step,
        A, B, C,
        x1, x2,
        y1, y2,
        color
    );
}


__host__ float signedAngle(
    float A1, float B1, 
    float A2, float B2
){
    float v1_x = B1;
    float v1_y = -A1;

    float v2_x = B2;
    float v2_y = -A2;

    float cross = v1_x * v2_y - v2_x * v1_y;
    float dot = v1_x * v2_x + v1_y * v2_y;

    float radian = atan2(cross, dot); //radian
    
    return radian;                                                                       
}

__host__ bool turning(
    float angle_d_up, float angle_d_down
) {
    int angle_up = (int)fabs(angle_d_up * 10);
    int angle_down = (int)fabs(angle_d_down * 10);

    int error_angle = abs(abs(angle_up) - abs(angle_down));
    //std::cout<<"error_angle: "<<error_angle<<endl;

    if (error_angle >= 30) return true;
    else return false;
}

__host__ float mapValue(
    long x,
    long in_min, long in_max,
    long out_min, long out_max
){
    long lim_min = min (in_min, in_max);
    long lim_max = max (in_min, in_max);

    if (x > lim_max)  x =  lim_max;
    if (x < lim_min)  x = lim_min;
    return (x - in_min) * (out_max - out_min)
           / (in_max - in_min)
           + out_min;
}



__host__ int findXlinear (
    float A_down, float B_down, float C_down
){
    if (A_down == 0) {
        return 0;
    }
    else {
        float x_linear = -(C_down + 480*B_down) / A_down;
        int x_linear_int = (int)(x_linear + 0.5f);
        //std::cout<<"x_linear_int: "<<x_linear_int<<endl;
        return x_linear_int;
    }
    
}

__global__ void SumDelta(
    const uint16_t* yellow_x_host, int N,
    int* sumdelta
){
    __shared__ int sum_delta[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    int delta = 0;
    if (idx < N) {
        delta = 355 - yellow_x_host[idx];
    }
    atomicAdd(sumdelta, delta);
}

__host__ int DeltaAvg(
    const vector<uint16_t> yellow_x_host
){
    int N = yellow_x_host.size();
    int block = 256;
    int grid = (N + block - 1) / block;

    int* sum_delta_d;
    uint16_t* yellow_x_device;
    
    cudaMalloc(&sum_delta_d, sizeof(int));
    cudaMalloc(&yellow_x_device, N * sizeof(uint16_t));

    cudaMemcpy(yellow_x_device, yellow_x_host.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemset(sum_delta_d, 0, sizeof(int));

    SumDelta<<<grid, block>>>(
        yellow_x_device, N,
        sum_delta_d
    );

    int sum_delta_h;
    cudaMemcpy(&sum_delta_h, sum_delta_d, sizeof(int), cudaMemcpyDeviceToHost);
    int delta_avg = sum_delta_h * 10 / N;

    cudaFree(yellow_x_device);
    cudaFree(sum_delta_d);

    return delta_avg;
}

float mapf(float x,
           float in_min, float in_max,
           float out_min, float out_max)
{
    return (x - in_min) * (out_max - out_min)
           / (in_max - in_min)
           + out_min;
}


__host__ void find_point_look (
    float A_up, float B_up, float C_up,
    float A_down, float B_down, float C_down,
    int inte, int y_inte,
    int xc, int yc,
    float ld,
    int &x_look, int &y_look
){
   y_look = yc - ld;
   if (y_inte <= 250 || inte == 0) {
    x_look = -(B_down * y_look + C_down) / A_down;
   }
   else if (y_inte >= 255 && inte == 1) {
    x_look = -(B_up * y_look + C_up) / A_up;
   }
   //std::cout<<"x_look: "<<x_look<<endl;
}

__host__ void find_angle_drive (
    int L, int Ld, int ob,
    float angle_down, float angle_up,
    int inte, int y_inte,
    int xc, int yc, int x_look, int y_look,
    int &angle_drive
) {

    float dx = x_look - xc;
    float dy = y_look - yc;
    float k;

    float alpha = atan2(dx, -dy);

    float delta_pp =  atan2(2.0f * L * sin(alpha), Ld);

    float theta;
    if (y_inte <= 270 || y_inte >= 470 || inte == 0) theta = angle_down;
    else if (y_inte >= 275 && y_inte <= 465 && inte == 1) theta = angle_up;
    if (ob == 2) k = 0.0f;
    else k = 3.5f;
    float delta = delta_pp + k * theta;

    int delta_int= (int)(delta * 1000);
    if (delta_int <= -2120) delta_int = -2120;
    else if (delta_int >= 2120) delta_int = 2120;

    angle_drive = mapValue(delta_int, -2120, 2120, 115, 60);

    //std::cout<<"delta_pp: "<<delta_pp<<endl;
    //std::cout<<"theta: "<<theta<<endl;
    //std::cout<<"delta: "<<delta<<endl;
    //std::cout<<"angle_drive: "<<angle_drive<<endl;
    //std::cout<<endl;

}

__global__ void lidar_point_xy(
    const int* angle, const int* dist,
    int* point_x, int* point_y, int n
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        float theta = (float)((angle[idx] + 550) * M_PI / (100.0f * 180.0f));
        point_x[idx] = (int)(dist[idx] * sinf(theta) / 10000); 
        point_y[idx] = (int)(dist[idx] * cosf(theta) /10000);
    }
}

__host__ void lidar_map(
    const std::vector<int>& angle_host,
    const std::vector<int>& dist_host,
    std::vector<int>& point_x,
    std::vector<int>& point_y
){
    int* angle_d;
    int* dist_d;
    int* point_x_d;
    int* point_y_d;

    int n = angle_host.size();
    if (n == 0) return;


    cudaMalloc(&angle_d, n*sizeof(int));
    cudaMalloc(&dist_d, n*sizeof(int));
    cudaMalloc(&point_x_d, n*sizeof(int));
    cudaMalloc(&point_y_d, n*sizeof(int));

    cudaMemcpy(angle_d, angle_host.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dist_d, dist_host.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    int blocks = 256;
    int grids = (n + blocks - 1) / blocks;

    lidar_point_xy<<<grids, blocks>>>(
        angle_d, dist_d,
        point_x_d, point_y_d, n
    );

    cudaDeviceSynchronize();

    point_x.resize(n);
    point_y.resize(n);


    cudaMemcpy(point_x.data(), point_x_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_y.data(), point_y_d, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(angle_d);
    cudaFree(dist_d);
    cudaFree(point_x_d);
    cudaFree(point_y_d);
}

__host__ void detect_object(
    const std::vector<int>& point_x_host, 
    const std::vector<int>& point_y_host, 
    std::vector<int>& x_front, std::vector<int>& dist_front,
    std::vector<int>& x_left, std::vector<int>& dist_left,
    std::vector<int>& x_right, std::vector<int>& dist_right
){

    x_front.clear();
    x_left.clear();
    x_right.clear();
    dist_front.clear();
    dist_left.clear();
    dist_right.clear();

    int x, y;
    int n = point_x_host.size();

    if (n <= 0) return;

    for (int i = 0; i < n; i++){
        x = point_x_host[i];
        y = point_y_host[i];

        if (x >= -10 && x <= 10 && y >= 0){
            x_front.push_back(x);
            dist_front.push_back(y);
        }

        else if (y >= -5 && y <= 5 && x <= 0){
            x_left.push_back(y);
            dist_left.push_back(x);
        }

        else if (y >= -5 && y <= 5 && x > 0){
            x_right.push_back(y);
            dist_right.push_back(x);
        }
    }

}

__host__ void Avoidance(
    std::vector<int>& x_front, std::vector<int>& dist_front,
    int& delta_ob,
    int& ob
){
    std::vector<int> x_object;
    int n_front = x_front.size();
    int x_ob_min, x_ob_max;
    int delta_ob_l, delta_ob_r;

    for (int i = 0; i < n_front; i++ ){
        if (dist_front[i] <= 25) {
            x_object.push_back(x_front[i]);
        }
    }

    int n_object = x_object.size();

    if (n_object >= 3){
        
        x_ob_min = *std::min_element(x_object.begin(), x_object.end());
        x_ob_max = *std::max_element(x_object.begin(), x_object.end()) - 2;
     
        
        delta_ob_l = -10 - x_ob_min;
        delta_ob_r = 10 - x_ob_max;

        

        int x_cen_ob = (x_ob_max + x_ob_min) / 2;
        int width_ob = x_ob_max - x_ob_min;
        //std::cout<<"x_cen_ob "<<x_cen_ob<<endl;
        //std::cout<<"x_ob_min "<<x_ob_min<<endl;
        //std::cout<<"x_ob_max "<<x_ob_max<<endl;

        if (abs(x_cen_ob) <= 2 && width_ob >= 7) {
            ob = 1;
            delta_ob = 0;
        }
        else if (abs(delta_ob_l) > abs(delta_ob_r) && abs(delta_ob_l) > 9){
            ob = 2;
            delta_ob = delta_ob_l;
        }

        else if (abs(delta_ob_l) > abs(delta_ob_r) && abs(delta_ob_l) <= 8){
            ob = 1;
            delta_ob = 0;
        }


        else if (abs(delta_ob_r) > abs(delta_ob_l) && abs(delta_ob_r) > 5){
            ob = 2;
            delta_ob = delta_ob_r;
        }
        else if (abs(delta_ob_r) > abs(delta_ob_l) && abs(delta_ob_r) <= 4){
            ob = 1;
            delta_ob = 0;
            std::cout<<"delta_ob_r "<<delta_ob_r<<endl;
        }


    }
    else if (n_object < 3) {
        delta_ob = 0;
        ob = 0;
        return;
    }
}

__host__ void Avoidance_left(
    const std::vector<int>& x_left,
    const std::vector<int>& dist_left,
    int& delta_left
){
    int n_left = dist_left.size();
    if (n_left < 3) {
        delta_left = 1000;   // xa an toàn
        return;
    }

    int sum  = 0;
    int cnt  = 0;

    for (int i = 0; i < n_left; i++) {
        if (dist_left[i] >= -15) {
            sum += dist_left[i];
            cnt++;
        }
    }

    if (cnt > 0)
        delta_left = sum / cnt;
    else
        delta_left = 1000;
}


__host__ void Avoidance_right(
    const std::vector<int>& x_right,
    const std::vector<int>& dist_right,
    int& delta_right
){
    int n_right = dist_right.size();
    if (n_right < 3) {
        delta_right = 1000;
        return;
    }

    int sum = 0;
    int cnt = 0;

    for (int i = 0; i < n_right; i++) {
        if (dist_right[i] <= 15) {
            sum += dist_right[i];
            cnt++;
        }
    }

    if (cnt > 0)
        delta_right = sum / cnt;
    else
        delta_right = 1000;
}

__device__ bool PointInQuad (
    uint32_t x, uint32_t y,
    uint4* pts
){
    uint32_t x0 = pts[0].x, y0 = pts[0].y;
    uint32_t x1 = pts[1].x, y1 = pts[1].y;  
    uint32_t x2 = pts[2].x, y2 = pts[2].y;
    uint32_t x3 = pts[3].x, y3 = pts[3].y;
    uint32_t x4 = pts[4].x, y4 = pts[4].y;
    uint32_t x5 = pts[5].x, y5 = pts[5].y;

    int32_t v_x[6] = {x1 - x0, x2 - x1, x3 - x2, x4 - x3, x5 - x4, x0 - x5};
    int32_t v_y[6] = {y1 - y0, y2 - y1, y3 - y2, y4 - y3, y5 - y4, y0 - y5};

    int32_t p_x[6] = {x - x0, x - x1, x - x2, x - x3, x - x4, x - x5};
    int32_t p_y[6] = {y - y0, y - y1, y - y2, y - y3, y - y4, y - y5};

    bool hasPos = false, hasNeg = false;
    int32_t cross;
    for (uint8_t i = 0; i < 6; i++) {
        cross = v_x[i]*p_y[i] - v_y[i]*p_x[i];
        if (cross > 0) hasPos = true;
        if (cross < 0) hasNeg = true;
        if (hasPos && hasNeg) return false;
    }
    return true;
}

__global__ void MaskQuadKernel (
    uchar* frame, uint32_t width, uint32_t height,
    size_t step, uint4* pts
    
){
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    bool inside = PointInQuad(x, y, pts);

    if (!inside) {
        uchar* row = frame + y * step;  
        row[x] = 0; 
    }
}

cuda::GpuMat MaskQuad (
    const cv::cuda::GpuMat& input,
    const std::array<cv::Point,6>& quad
){
    uint4 h_pts[6];
    for (int i=0;i<6;i++) h_pts[i] = make_uint4(quad[i].x, quad[i].y, 0, 0);

    uint4* d_pts;
    cudaMalloc(&d_pts, sizeof(h_pts));
    cudaMemcpy(d_pts, h_pts, sizeof(h_pts), cudaMemcpyHostToDevice);

    cv::cuda::GpuMat output;
    input.copyTo(output);

    dim3 block(16,16);
    dim3 grid((input.cols+15)/16, (input.rows+15)/16);

    MaskQuadKernel<<<grid, block>>>(
        output.ptr<uchar>(), 
        input.cols, 
        input.rows, 
        output.step, 
        d_pts
    );

    cudaFree(d_pts);

    return output; 
}

// pipeline và FPSCounter (giữ nguyên)
string gstreamer_pipeline(
    uint8_t sensor_id = 0,
    uint16_t capture_width = 1280,
    uint16_t capture_height = 720,
    uint16_t display_width = 640,
    uint16_t display_height = 480,
    uint8_t framerate = 50,
    uint8_t flip_method = 0
) {
    return "nvarguscamerasrc sensor-id=" + to_string(sensor_id) + " ! "
           "video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) +
           ", height=(int)" + to_string(capture_height) +
           ", framerate=(fraction)" + to_string(framerate) + "/1 ! "
           "nvvidconv flip-method=" + to_string(flip_method) + " ! "
           "video/x-raw, width=(int)" + to_string(display_width) +
           ", height=(int)" + to_string(display_height) +
           ", format=(string)BGRx ! "
           "videoconvert ! "
           "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=false";
}

class FPSCounter {
    int64 lastTick;
public:
    FPSCounter() { lastTick = cv::getTickCount(); }
    double newFrame() {
        int64 now = cv::getTickCount();
        double dt = (now - lastTick) / cv::getTickFrequency();
        lastTick = now;
        return (dt > 0) ? 1.0 / dt : 0.0;
    }
};

// ---------------- main ----------------
int main() {
    // Khuyến nghị: nếu muốn ổn định khi rút/cắm, dùng đường dẫn trong /dev/serial/by-id/ thay cho /dev/ttyUSB0
    const char* port = "/dev/ttyUSB0";
    std::atomic<int> afd(openSerial(port, B115200));
    if (afd.load() < 0) {
        cerr << "[Serial] Không mở được " << port << " ngay từ đầu — sẽ thử reconnect tự động.\n";
    } else {
        cout << "[Serial] Đã mở " << port << "\n";
    }
    // Fix encoding
    setlocale(LC_ALL, "C.UTF-8");
    
    printf("Receiving LIDAR data...\n");
    printf("Format: angle distance\n\n");
    
    char line[256];
    int count = 0;

    string pipeline = gstreamer_pipeline();
    cout << "PIPELINE:\n" << pipeline << endl;
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) { cerr << "Không mở được camera\n"; running = false; return -1; }

    // shared memory
    int fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }

    shm_frame = (Frame*)mmap(nullptr, sizeof(Frame), PROT_READ, MAP_SHARED, fd, 0);
    if (shm_frame == MAP_FAILED) { perror("mmap"); return 1; }

    std::cout << "Reading LiDAR frames from shared memory..." << std::endl;


    std::cout << "Reading LiDAR frame from shared memory...\n";

    cv::Mat frame, result_frame;
    cuda::GpuMat gpuFrame, gpuGray, gpuBlur, gpuCanny, gpuMasked, test_mask;
    FPSCounter fpsCounter;
    double fps; int delta_r = 0, delta_l = 0;

    // start serial thread (truyền tham chiếu atomic fd)
    std::thread th(serialThread, std::ref(afd), port, B115200);

    Ptr<cuda::Filter> blurFilter = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(5,5));
    Ptr<cuda::CannyEdgeDetector> Canny = cuda::createCannyEdgeDetector(50.0, 150.0, 3, false);

    // ROI tương ứng với 640x480
    std::array<cv::Point,6> quad = {
        cv::Point(170, 250),
        cv::Point(500,250),
        cv::Point(640,250),
        cv::Point(640,480),
        cv::Point(30,480),
        cv::Point(30,250)
    };

    std::vector<uint16_t>yellow_x_up_host;
    std::vector<uint16_t>yellow_y_up_host;
    std::vector<uint16_t>yellow_x_down_host;
    std::vector<uint16_t>yellow_y_down_host;

    std::vector<int> frame_LD_host_angle;
    std::vector<int> frame_LD_host_distance;
    std::vector<int> point_x_host;
    std::vector<int> point_y_host;

    std::vector<int> x_front; std::vector<int> dist_front;
    std::vector<int> x_left; std::vector<int> dist_left;
    std::vector<int> x_right; std::vector<int> dist_right;
    

    float A_up, B_up, C_up;
    float A_down, B_down, C_down;
    int x1_up, x2_up, x1_down, x2_down, y1_up, y2_up, y1_down, y2_down;
    int inte, x_cen = 340, T;
    float A_center = -1, B_center = 0, C_center = -355;
    float angle_d_up, angle_d_down;
    int x_linear, x_look, y_look,  angle_drive;
    float y_inte, x_inte;
    int delta_ob, ob, delta_left, delta_right;
    int cout;
    uint8_t Auto = 0;

    static float A_up_prev = 0, B_up_prev = 0, C_up_prev = 0;
    static float A_down_prev = 0, B_down_prev = 0, C_down_prev = 0;
    static float angle_drive_prev = 0;



    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        int cols = frame.cols;
        int rows = frame.rows;


        if (Auto == 0) {

            gpuFrame.upload(frame);
            cv::cuda::cvtColor(gpuFrame, gpuGray, cv::COLOR_BGR2GRAY);
            blurFilter->apply(gpuGray, gpuBlur);
            test_mask = MaskQuad(gpuBlur, quad);
            Canny->detect(gpuBlur, gpuCanny);
            gpuMasked = MaskQuad(gpuCanny, quad);

        }
        
        if(shm_frame->count != last_count && shm_frame->count > 0) {
            frame_LD_host_angle.resize(shm_frame->count);
            frame_LD_host_distance.resize(shm_frame->count);

            std::memcpy(frame_LD_host_angle.data(), shm_frame->angle, shm_frame->count * sizeof(int));
            std::memcpy(frame_LD_host_distance.data(), shm_frame->distance, shm_frame->count * sizeof(int));

            last_count = shm_frame->count;
        }

        lidar_map(
            frame_LD_host_angle, frame_LD_host_distance,
            point_x_host, point_y_host);

        detect_object(
            point_x_host, point_y_host,
            x_front, dist_front,
            x_left, dist_left,
            x_right, dist_right
        );

        Avoidance(
            x_front, dist_front,
            delta_ob, ob
        );

        //std::cout<<"delta_ob: "<<delta_ob<<endl;
        //if (ob == 0) std::cout<<"khong co vat can phia truoc"<<endl;
        //std::cout<<"Co vat can !!"<<" xin hay qua left"<<endl;
        //else if (ob == 2 && delta_ob > 0) std::cout<<"Co vat can !!"<<" xin hay qua right"<<endl;
        //else if (ob == 1) std::cout<<"xe bi chan !!!!!"<<endl;


        if (Auto == 0){
            uint16_t len_up = 0;
            uint16_t len_down = 0;
        
            DrawTwoLaneCenterGPU(
                gpuFrame, gpuMasked,
                yellow_x_up_host, yellow_y_up_host,
                yellow_x_down_host, yellow_y_down_host,
                len_up, len_down
            );

            if (ob == 2) {
            
            if (delta_ob > 0 && delta_ob < 14){
                T = 1;
                x_cen = mapValue(delta_ob, 12, 10, 360, 200);
            }
            else if (delta_ob < 0 && delta_ob > -16){
                T = -1;
                x_cen = mapValue(delta_ob, -12, -10, 360 , 480);
            }
            else if (delta_ob >= 14){
                x_cen = 480;

            }
            else if (delta_ob <= -16){
                x_cen = 200;

            }

            scanLaneByDirection(
                gpuMasked, delta_ob, x_cen,
                yellow_x_up_host, yellow_y_up_host,
                yellow_x_down_host, yellow_y_down_host,
                len_up, len_down
            );
        }
            else if (ob == 1 || ob == 0){
            x_cen = 360;
        }


            //std::cout<<"ob: "<<ob<<endl;
            //std::cout<<"delta_ob: "<<delta_ob<<endl;
            //std::cout<<"x_cen: "<<x_cen<<endl;


            bool has_up   = len_up   > 2;
            bool has_down = len_down > 2;
 
            //================ FIT LINE (CÓ FALLBACK) =================
            if (has_up) {
            Linear(
                yellow_x_up_host,
                yellow_y_up_host,
                A_up, B_up, C_up
            );
        } 
            else {
            A_up = A_up_prev;
            B_up = B_up_prev;
            C_up = C_up_prev;
        }

            if (has_down) {
            Linear(
                yellow_x_down_host,
                yellow_y_down_host,
                A_down, B_down, C_down
            );
        } 
            else {
            A_down = A_down_prev;
            B_down = B_down_prev;
            C_down = C_down_prev;
        }

            //================ LƯU TRẠNG THÁI =================
            if (has_up) {
            A_up_prev = A_up;
            B_up_prev = B_up;
            C_up_prev = C_up;
        }
            if (has_down) {
            A_down_prev = A_down;
            B_down_prev = B_down;
            C_down_prev = C_down;
        }

            //================ INTERSECTION =================
            intersection(
                rows, cols,
                A_up, B_up, C_up,
                A_down, B_down, C_down, 
                y_inte, x_inte, inte
            );

            //================ CHỌN ĐIỂM VẼ =================
            if (inte == 1) {
            if (has_up) {
                x1_up = yellow_x_up_host[0];
                y1_up = yellow_y_up_host[0];
                x2_up = (int)(x_inte + 0.5f);
                y2_up = (int)(y_inte + 0.5f);
            }

            if (has_down) {
                x1_down = (int)(x_inte + 0.5f);
                y1_down = (int)(y_inte + 0.5f);
                x2_down = yellow_x_down_host[len_down - 1];
                y2_down = yellow_y_down_host[len_down - 1];
            }
        }
            else {
            if (has_up) {
                x1_up = yellow_x_up_host[0];
                y1_up = yellow_y_up_host[0];
                x2_up = yellow_x_up_host[len_up - 1];
                y2_up = yellow_y_up_host[len_up - 1];
            }

            if (has_down) {
                x1_down = yellow_x_down_host[0];
                y1_down = yellow_y_down_host[0];
                x2_down = yellow_x_down_host[len_down - 1];
                y2_down = yellow_y_down_host[len_down - 1];
            }
        }

            //================ VẼ LINE =================
            if (has_up) {
            drawLinearOnGpu(
                gpuFrame,
                A_up, B_up, C_up,
                x1_up, x2_up,
                y1_up, y2_up,
                make_uchar3(0, 0, 255)
            );
        }

            if (has_down) {
            drawLinearOnGpu(
                gpuFrame,
                A_down, B_down, C_down,
                x1_down, x2_down,
                y1_down, y2_down,
                make_uchar3(0, 0, 255)
            );
        }

            //================ TÍNH GÓC =================
            angle_d_up = signedAngle(
                A_center, B_center,
                A_up, B_up
            );

            angle_d_down = signedAngle(
                A_center, B_center,
                A_down, B_down
            );

            x_linear = findXlinear(A_down, B_down, C_down);

            //================ LOOK AHEAD =================
            find_point_look(
                A_up, B_up, C_up,
                A_down, B_down, C_down,
                inte, y_inte,
                x_cen, 480, 60,
                x_look, y_look
            );

            //================ GÓC LÁI CUỐI =================
            find_angle_drive(
                60, 60, ob,
                angle_d_down, angle_d_up,
                inte, y_inte,
                x_cen, 480,
                x_look, y_look,
                angle_drive
            );
        
            //std::cout<<"angle_drive: "<< angle_drive<<endl;
            gpuFrame = drawLineCenter(gpuFrame, x_cen);
            gpuFrame.download(result_frame);
            fps = fpsCounter.newFrame();
            cv::putText(result_frame, "FPS: " + to_string((int)fps),
            {20,30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        }

        else if (Auto == 1){
            

            if (delta_ob < 0) {
                angle_drive = mapValue(delta_ob, -17, -12, 90, 120);
            }
            else if (delta_ob > 0) {
                angle_drive = mapValue(delta_ob, 18, 10, 90, 60);
            }
            else if (delta_ob == 0) {
                angle_drive = 90;
            }
            //std::cout<<"delta_ob: "<<delta_ob<<endl;
            //std::cout<<"angle_drive: "<<angle_drive<<endl;
            //std::cout<<"ob: "<<ob<<endl;

            result_frame = frame;
        }
        
        
        cudaDeviceSynchronize();

        cv::imshow("CAMERA", result_frame);

        // chỉ push message vào hàng đợi, thread serial sẽ gửi và reconnect khi cần
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push("a" + std::to_string(angle_drive) + "b" + std::to_string(ob) + "\n");
        }

        


        if (cv::waitKey(1) >= 0) break;
    }

    // cleanup
    munmap(shm_frame, sizeof(Frame));
    close(fd);
    running = false;
    th.join();
    int final_fd = afd.load();
    if (final_fd >= 0) close(final_fd);
    return 0;

}
