#include <iostream>
#include <stdio.h>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

double* generateRandomVector(double* randomVector, int n) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-10, 10);

    // Fill the vector with random integers
    for (int i = 0; i < n; ++i) {
        randomVector[i] = distrib(gen);
    }

    return randomVector;
}

void printVector(double* vector, int n) {
    std::cout << "[" << " ";
    for (int i = 0; i < n; i++) {
        std::cout << vector[i] << ", ";
    }
    std::cout << "]" << " ";
    std::cout << std::endl;
}

double calculateResidual(double* result_gpu, double* result_cpu, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += std::abs(result_gpu[i] - result_cpu[i]);
    }

    return sum / size;
}

__global__ void addVectors(double* c, double* a, double* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}


//Function used for correct cases
void vectorAddition(int blockSize, int threadSize) {
    int sizeOfVector = blockSize * threadSize;
    int memSize = sizeOfVector * sizeof(double);

    double* a = (double*)malloc(memSize);
    generateRandomVector(a, sizeOfVector);
    double* b = (double*)malloc(memSize);
    generateRandomVector(b, sizeOfVector);
    double* d_a, * d_b, * d_c;

    double* c_sequential = (double*)malloc(memSize);
    double* c_gpu = (double*)malloc(memSize);

    //Starting sequential CPU execution of vector addition
    auto start_fun_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < sizeOfVector; i++) {
        c_sequential[i] = a[i] + b[i];
    }

    auto end_fun_time = std::chrono::high_resolution_clock::now();
    auto duration_sequential = std::chrono::duration_cast<std::chrono::microseconds>(end_fun_time - start_fun_time);

    //Starting GPU execution of vector addition
    cudaMalloc((void**)&d_a, memSize);
    cudaMalloc((void**)&d_b, memSize);
    cudaMalloc((void**)&d_c, memSize);

    auto start_mem_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    start_fun_time = std::chrono::high_resolution_clock::now();

    addVectors << <blockSize, threadSize >> > (d_c, d_a, d_b);
    cudaDeviceSynchronize();

    end_fun_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(c_gpu, d_c, memSize, cudaMemcpyDeviceToHost);

    auto end_mem_time = std::chrono::high_resolution_clock::now();

    auto duration_mem_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_mem_time - start_mem_time);
    auto duration_fun_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_fun_time - start_fun_time);

    double residual = calculateResidual(c_gpu, c_sequential, sizeOfVector);

    //Freeing memory
    free(a);
    free(b);
    free(c_sequential);
    free(c_gpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //Printing out results
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Thread Size: " << threadSize << std::endl;
    std::cout << "Size Of Vector: " << sizeOfVector << std::endl;

    //Checking if any error occurred in CPU processing
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n" << std::endl;
    }

    std::cout << "CPU Sequential Execution Time: " << duration_sequential.count() << " microseconds" << std::endl;
    std::cout << "GPU Time with Memory Copy Calls: " << duration_mem_gpu.count() << " microseconds" << std::endl;
    std::cout << "GPU Time (only function calls): " << duration_fun_gpu.count() << " microseconds\n" << std::endl;

    std::cout << "Residual: " << residual << std::endl;
    std::cout << "******************************************************\n" << std::endl;
}

//Overloading vectorAddition to print out the vectors for edge cases/error scenarios
void vectorAddition(int blockSize, int threadSize, int sizeOfVector) {
    int memSize = sizeOfVector * sizeof(double);

    double* a = (double*)malloc(memSize);
    generateRandomVector(a, sizeOfVector);
    double* b = (double*)malloc(memSize);
    generateRandomVector(b, sizeOfVector);
    double* d_a, * d_b, * d_c;

    double* c_sequential = (double*)malloc(memSize);
    double* c_gpu = (double*)malloc(memSize);

    //Starting sequential CPU execution of vector addition
    auto start_fun_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < sizeOfVector; i++) {
        c_sequential[i] = a[i] + b[i];
    }

    auto end_fun_time = std::chrono::high_resolution_clock::now();
    auto duration_sequential = std::chrono::duration_cast<std::chrono::microseconds>(end_fun_time - start_fun_time);

    //Starting GPU execution of vector addition
    cudaMalloc((void**)&d_a, memSize);
    cudaMalloc((void**)&d_b, memSize);
    cudaMalloc((void**)&d_c, memSize);

    auto start_mem_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    start_fun_time = std::chrono::high_resolution_clock::now();

    addVectors << <blockSize, threadSize >> > (d_c, d_a, d_b);
    cudaDeviceSynchronize();

    end_fun_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(c_gpu, d_c, memSize, cudaMemcpyDeviceToHost);

    auto end_mem_time = std::chrono::high_resolution_clock::now();

    auto duration_mem_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_mem_time - start_mem_time);
    auto duration_fun_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_fun_time - start_fun_time);

    double residual = calculateResidual(c_gpu, c_sequential, sizeOfVector);

    //Printing out results
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Thread Size: " << threadSize << std::endl;
    std::cout << "Size Of Vector: " << sizeOfVector << std::endl;

    //Checking if any error occurred in CPU processing
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n" << std::endl;
    }

    std::cout << "CPU Sequential Execution Time: " << duration_sequential.count() << " microseconds" << std::endl;
    std::cout << "GPU Time with Memory Copy Calls: " << duration_mem_gpu.count() << " microseconds" << std::endl;
    std::cout << "GPU Time (only function calls): " << duration_fun_gpu.count() << " microseconds\n" << std::endl;

    std::cout << "Residual: " << residual << "\n" << std::endl;

    std::cout << "Vector1: ";
    printVector(a, sizeOfVector);
    std::cout << "Vector2: ";
    printVector(b, sizeOfVector);
    std::cout << "Result: ";
    printVector(c_gpu, sizeOfVector);
    std::cout << "******************************************************\n" << std::endl;

    //Freeing memory
    free(a);
    free(b);
    free(c_sequential);
    free(c_gpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/*To compile and run this file, create a new CUDA C++ project in Visual Studio.
Next, add this file to the project.
Build your solution, and run.*/
int main() {
    //Initializing all possible block sizes
    int blockSizes[10] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    int j;

    //Executing correct cases
    for (int i = 0; i < 10; i++) {
        //ThreadSizes will range from 8 to 1024
        j = 8;
        while (j <= 1024) {
            vectorAddition(blockSizes[i], j);
            j *= 2;
        }
    }

    //Trying odd cases
    vectorAddition(1, 2048, 2048);
    vectorAddition(1, 1, 10);
    vectorAddition(1, 10, 5);
    vectorAddition(1, 5, 10);
    vectorAddition(1, 33, 33);

    return 0;
}