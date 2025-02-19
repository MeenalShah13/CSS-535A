#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <future>
#include <mkl.h>
#include <cmath>

std::vector<double> generateRandomMatrix(int n) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-10, 10);

    // Create the matrix
    std::vector<double> matrix(n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = distrib(gen); // Row-major indexing as C++ is row-major
        }
    }

    return matrix;
}

double calculateResidual(std::vector<double> calculatedResult, std::vector<double> blasMatrix) {
    int size = std::sqrt(calculatedResult.size());
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sum += std::abs(calculatedResult[i * size + j] - blasMatrix[i * size + j]);
        }
    }
    return sum / (size * size);
}

void assignmentSolution(std::vector<int>& sizeList) {

    //main program
    for (int size : sizeList) {

        double flop = 2 * std::pow(size, 3);
        double memorySteps = std::pow(size, 3) + 3 * std::pow(size, 2);
        double theoreticalPeakPerCycle = flop / memorySteps;
        double theoreticalPeak = theoreticalPeakPerCycle * 16 * 2.2 * 1000000000; // theoreticalPeakPerCycle * numberOfCores * ClockSpeed
        
        std::vector<std::future<void>> futures;
        double temp;

        std::vector<double> matrix1 = generateRandomMatrix(size);
        std::vector<double> matrix2 = generateRandomMatrix(size);
        std::vector<double> result_sequential(size * size);
        std::vector<double> correctResult(size * size);

        //start time for sequential gemv calculation
        auto start_time = std::chrono::high_resolution_clock::now();

        //Naive Multiplication
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                temp = 0;
                for (int k = 0; k < size; k++) {
                    temp += matrix1[i * size + k] * matrix2[k * size + j];
                }
                result_sequential[i * size + j] = temp;
            }
        }

        //end time for gemv calculation
        auto end_time = std::chrono::high_resolution_clock::now();
        auto sequential_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        //Checking result against BLAS implementation and calculating residual
        start_time = std::chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, matrix1.data(), size, matrix2.data(), size, 0.0, correctResult.data(), size);
        end_time = std::chrono::high_resolution_clock::now();
        auto mkl_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        double residual_sequential = calculateResidual(result_sequential, correctResult);

        double flops_sequential = (flop * 1000) / sequential_duration; //Dividing by 1000 to convert to seconds
        double flops_mkl = (flop * 1000) / mkl_duration; //Dividing by 1000 to convert to seconds

        double efficiency_sequential = flops_sequential / theoreticalPeak * 100;
        double efficiency_mkl = flops_mkl / theoreticalPeak * 100;

        std::cout << "Size: " << size << std::endl;
        std::cout << "Sequential Implementation:" << std::endl;
        std::cout << "Time taken: " << sequential_duration << " milliseconds" << std::endl;
        std::cout << "Flops: " << flops_sequential << std::endl;
        std::cout << "Efficiency: " << efficiency_sequential << std::endl;
        std::cout << "Residual: " << residual_sequential << "\n" << std::endl;
        std::cout << "BLAS Implementation:" << std::endl;
        std::cout << "Time taken: " << mkl_duration << " milliseconds" << std::endl;
        std::cout << "Flops: " << flops_mkl << std::endl;
        std::cout << "Efficiency: " << efficiency_mkl << std::endl;
        std::cout << "****************************\n" << std::endl;
    }
}

int main() {
    std::vector<int> sizeList = {10, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000 };
    assignmentSolution(std::ref(sizeList));
    return 0;
}