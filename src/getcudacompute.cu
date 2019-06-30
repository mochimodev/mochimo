#include <stdio.h>

int main() {
        cudaDeviceProp prop;
        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);

        int *found_versions = (int*)malloc(num_devices * sizeof(int));
        int num_versions = 0;

        for (int i = 0; i < num_devices; i++) {
                found_versions[i] = 0;
                cudaGetDeviceProperties(&prop, i);

                int majmin = prop.major * 1000 + prop.minor;
                int found = 0;
                for (int j = 0; j < num_versions; j++) {
                        if (found_versions[j] == majmin) {
                                // Already found
                                found = 1;
                                break;
                        }
                }
                if (!found) {
                        printf("--generate-code arch=compute_%d%d,code=sm_%d%d ",
                                        prop.major, prop.minor, prop.major, prop.minor);
                        found_versions[num_versions++] = majmin;
                }
        }

        free(found_versions);
}
