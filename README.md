# Game of Life using CUDA

This is a simple implementation of the [Conway's Game of Life](
http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) using CUDA. The intention
of this code is just for didactic purposes on the teaching of GPU programming.
This example is focussed particularly in the concept of shared memory and the
improvement in the running time.

To run this example you need:

- cmake: For compilation
- OpenCV: For image display
- CUDA: For obvious reasons :)

To execute it, enter in the top directory and do the following:

```bash
mkdir build
cd build
cmake ..
make
./gol
```

Note that it is very important to specify the CUDA capability of your device in
the **CUDA_NVCC_FLAGS** variable inside the CMakeLists.txt file.
