#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include "/home/nanob01/rplidar_vscode_project/data.h"

int main() {
    // Fix encoding
    setlocale(LC_ALL, "C.UTF-8");
    
    printf("Receiving LIDAR data...\n");
    printf("Format: angle distance\n\n");
    
    char line[256];
    int count = 0;
    
    while (1) {
        if (fgets(line, sizeof(line), stdin) != NULL) {
            float angle, distance;
        
        
            if (sscanf(line, "%f %f", &angle, &distance) == 2) {
                count++;
            
                // In ra console (không unicode)
            
            
                // In thông tin m?i 10 ði?m
                if (count % 10 == 0) {
                    printf("--- Received %d points ---\n", count);
                    
                }
            }
        }
    }
    
    printf("\nTotal points: %d\n", count);
    return 0;
}