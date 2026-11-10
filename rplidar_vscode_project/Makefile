CC = g++
CFLAGS = -I/usr/local/include -std=c++11 -O2
LDFLAGS = -L/usr/local/lib -lrplidar -lpthread
TARGET = lidar_single

all: $(TARGET)

$(TARGET): main.cpp
$(CC) $(CFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
rm -f $(TARGET)

run: $(TARGET)
sudo ./$(TARGET)

.PHONY: all clean run
