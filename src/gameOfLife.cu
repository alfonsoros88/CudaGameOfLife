#include <cuda_runtime.h>
#include "gameOfLife.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void gameOfLifeIt(unsigned char* d_dst, unsigned char* d_buff, const size_t width, const size_t height) {
    extern __shared__ unsigned char board_sh[];

    size_t glob_x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t glob_y = blockDim.y * blockIdx.y + threadIdx.y;
    size_t glob_idx = glob_y * width + glob_x;

    size_t index = blockDim.x * threadIdx.y + threadIdx.x;

    int share_width = blockDim.x + 2;
    int share_height = blockDim.y + 2;
    int share_size = share_width * share_height;

    bool isActive = (glob_x < width && glob_y < height);
    
    // Copy board to shared memory
    for (int share_idx = index; share_idx < share_size; share_idx += (blockDim.x * blockDim.y)) {

        int x_img = (blockDim.x * blockIdx.x - 1) + (share_idx % share_width);
        int y_img = (blockDim.y * blockIdx.y - 1) + (share_idx / share_width);

        if (x_img < 0) {
            x_img = width - 1;
        } else if (x_img > width - 1) {
            x_img = 0;
        }

        if (y_img < 0) {
            y_img = height - 1;
        } else if (y_img > height - 1) {
            y_img = 0;
        }

        board_sh[share_idx] = d_dst[width * y_img + x_img];
    }

    __syncthreads();

    if (isActive) {

        unsigned char me = board_sh[share_width * (threadIdx.y + 1) + threadIdx.x + 1];
   
        int count = 0;
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                count += board_sh[share_width * (threadIdx.y + 1 + j) + threadIdx.x + 1 + i];
            }
        } 
        count -= me;

        // Game of life rules
        if (me == 1) {
            if (count < 2) {
                d_buff[glob_idx] = 0;
            }
            else if (count < 4) {
                d_buff[glob_idx] = 1;
            }
            else {
                d_buff[glob_idx] = 0;
            }
        } else {
            if (count == 3) {
                d_buff[glob_idx] = 1;
            }
        }
    }
}

void runGameOfLife(unsigned char* d_dst, unsigned char* d_buff, const size_t width, const size_t height) {
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    size_t shared_size = (threads.x + 2) * (threads.y + 2);

    gameOfLifeIt<<<threads, grid, shared_size>>>(d_dst, d_buff, width, height);
    cudaMemcpy(d_dst, d_buff, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
}
