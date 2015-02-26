#include <iostream>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cuda_runtime.h>

#include "gameOfLife.h"

#define BOARD_SIZE_X 100
#define BOARD_SIZE_Y 100
#define CIRCLE_RADIUS 5

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void displayBoard(const cv::Mat& board, cv::Mat& image, const int radius) {
    for (int i = 0, x = radius; i < BOARD_SIZE_X; i++, x += 2 * radius) {
        for (int j = 0, y = radius; j < BOARD_SIZE_Y; j++, y += 2 * radius) {
            cv::Point2i p(x, y);
            if (board.at<unsigned char>(i, j) == 1) {
                cv::circle(image, p, radius, 255, CV_FILLED, 8, 0);
            } else {
                cv::circle(image, p, radius, 0, CV_FILLED, 8, 0);
            }
        }
    }
}

int main(int argc, const char *argv[])
{
    unsigned char* d_dst;
    unsigned char* d_buff;
    cv::Mat board = cv::Mat::zeros(BOARD_SIZE_X, BOARD_SIZE_Y, CV_8UC1);
    cv::Mat image = cv::Mat::zeros(2 * CIRCLE_RADIUS * BOARD_SIZE_X, 2 * CIRCLE_RADIUS * BOARD_SIZE_Y, CV_8UC1);

    // Initialize the board randomly
    srand(time(NULL));
    for (int i = 0; i < BOARD_SIZE_X; i++) {
        for (int j = 0; j < BOARD_SIZE_Y; j++) {
            board.at<unsigned char>(i, j) = rand() % 2;
        }
    }

    // Initialize device board
    cudaCheckError(cudaMalloc((void**)&d_dst, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc((void**)&d_buff, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char)));
    cudaCheckError(cudaMemcpy(d_dst, board.data, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char), cudaMemcpyHostToDevice));

    cv::namedWindow("Game of Life", CV_WINDOW_AUTOSIZE);

    int key;
    while (key = cv::waitKey(30)) {
        runGameOfLife(d_dst, d_buff, BOARD_SIZE_X, BOARD_SIZE_Y);
        cudaCheckError(cudaMemcpy(board.data, d_dst, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        displayBoard(board, image, CIRCLE_RADIUS);
        cv::imshow("Game of Life", image);

        if (key != -1) break;
    }

    cudaCheckError(cudaFree(d_dst));
    cudaCheckError(cudaFree(d_buff));

    return 0;
}
