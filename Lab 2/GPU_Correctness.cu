//#include <iostream>
//#include <stdio.h>
//#include <random>
//#include <chrono>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//
//double* generateRandomVector(double* randomVector, int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Fill the vector with random integers
//    for (int i = 0; i < n; ++i) {
//        randomVector[i] = distrib(gen);
//    }
//
//    return randomVector;
//}
//
//void printVector(double* vector, int n) {
//    std::cout << "[" << " ";
//    for (int i = 0; i < n; i++) {
//        std::cout << vector[i] << ", ";
//    }
//    std::cout << "]" << " ";
//    std::cout << std::endl;
//}
//
//__global__ void addVectors(double* c, double* a, double* b) {
//    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
//}
//
//int main() {
//    int size = 10;
//    int memSize = size * sizeof(double);
//    double* a = (double*)malloc(memSize);
//    double* b = (double*)malloc(memSize);
//    for (int i = 0; i < size; i++) {
//        a[i] = i;
//        b[i] = i;
//    }
//
//    double* c = (double*)malloc(memSize);
//    double* d_a, * d_b, * d_c;
//
//    cudaMalloc((void**)&d_a, memSize);
//    cudaMalloc((void**)&d_b, memSize);
//    cudaMalloc((void**)&d_c, memSize);
//
//    auto start_mem_time = std::chrono::high_resolution_clock::now();
//
//    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
//
//    auto start_fun_time = std::chrono::high_resolution_clock::now();
//
//    addVectors <<<1, size>>> (d_c, d_a, d_b);
//    cudaDeviceSynchronize();
//
//    auto end_fun_time = std::chrono::high_resolution_clock::now();
//
//    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
//
//    auto end_mem_time = std::chrono::high_resolution_clock::now();
//
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//    }
//
//    auto duration_mem = std::chrono::duration_cast<std::chrono::microseconds>(end_mem_time - start_mem_time);
//    auto duration_fun = std::chrono::duration_cast<std::chrono::microseconds>(end_fun_time - start_fun_time);
//
//    std::cout << "Time taken with Memory Copy: " << duration_mem.count() << " microseconds" << std::endl;
//    std::cout << "Time taken (only function calls): " << duration_fun.count() << " microseconds" << std::endl;
//    std::cout << "Matrix a: ";
//    printVector(a, size);
//    std::cout << "Matrix b: ";
//    printVector(b, size);
//    std::cout << "Matrix c: ";
//    printVector(c, size);
//
//    free(a);
//    free(b);
//    free(c);
//
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);
//
//    return 0;
//}