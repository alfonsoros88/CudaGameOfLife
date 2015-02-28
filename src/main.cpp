#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cuda_runtime.h>

#include <gameOfLife.h>
#include <BoardVisualization.hpp>

#define BOARD_SIZE_X 200
#define BOARD_SIZE_Y 200
#define CIRCLE_RADIUS 2

#define cudaCheckError(ans) { cudaCheckErrors((ans), __FILE__, __LINE__); }
inline void cudaCheckErrors(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initBoardAtRandom(cv::Mat& board) {
    srand(time(NULL));
    for (int i = 0; i < board.cols; i++) {
        for (int j = 0; j < board.rows; j++) {
            board.at<unsigned char>(i, j) = static_cast<unsigned char>(rand() % 2);
        }
    }
}


int main(int argc, const char *argv[])
{
    unsigned char* d_src;
    unsigned char* d_dst;

    cv::Mat board = cv::Mat::zeros(BOARD_SIZE_X, BOARD_SIZE_Y, CV_8UC1);
    BoardVisualization viewer(BOARD_SIZE_X, BOARD_SIZE_Y, CIRCLE_RADIUS);

    // Initialize the board randomly
    initBoardAtRandom(board);

    // Initialize device memory
    cudaCheckError(cudaMalloc((void**)&d_src, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc((void**)&d_dst, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char)));
    cudaCheckError(cudaMemcpy(d_dst, board.data, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int key;
    while (key = cv::waitKey(10)) {

        // Calling the kernel
        runGameOfLifeIteration(d_src, d_dst, BOARD_SIZE_X, BOARD_SIZE_Y);
        cudaCheckError(cudaMemcpy(board.data, d_dst, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(d_src, d_dst, BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

        /** This is just for display. You should not touch this.  **/
        viewer.displayBoard(board);
        if (key != -1) break;
    }
 
    // Free device memory
    cudaCheckError(cudaFree(d_src));
    cudaCheckError(cudaFree(d_dst));

    return 0;
}
