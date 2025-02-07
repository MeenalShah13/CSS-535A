#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <future>
#include <mkl.h>
#include <fstream>

static std::vector<double> generateRandomMatrix(int n) {
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

static std::vector<double> generateRandomVector(int n) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-10, 10);

    // Create a vector of the specified size
    std::vector<double> randomVector(n);

    // Fill the vector with random integers
    for (int i = 0; i < n; ++i) {
        randomVector[i] = distrib(gen);
    }

    return randomVector;
}

// Non-Optimized matrix-vector multiplication function
static double process_row(int row_number, int size, const std::vector<double>& vector, const std::vector<double>& matrix) {
    double temp = 0.0;
    for (size_t j = 0; j < size; ++j) {
        temp += vector[j] * matrix[row_number * size + j];
    }
    return temp;
}

static double calculateResidual(std::vector<double> vector, std::vector<double> blasVector) {
    int size = vector.size();
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += std::abs(vector[i] - blasVector[i]);
    }
    return sum / size;
}

void assignmentSolution(std::vector<int>& sizeList) {
    //Handling writing results into a csv
    std::ofstream outfile("Data.csv", std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: Data.csv" << std::endl;
    }
    std::vector<std::string> header = { "Size", "Sequential Time in microseconds", "Sequential Flops", "Sequential Residual", "Parallel Time in microseconds", "Parallel Flops", "Parallel Residual", "BLAS Time in microseconds", "BLAS Flops"};
    for (size_t i = 0; i < header.size(); ++i) {
        outfile << header[i];
        if (i < header.size() - 1) outfile << ","; // Comma separator
    }
    outfile << std::endl; // New line for header

    //main program
    for (int size : sizeList) {
        double flops = 2 * size * size;
        std::vector<std::future<double>> futures;
        double temp;

        std::vector<double> vector = generateRandomVector(size);
        std::vector<double> matrix = generateRandomMatrix(size);
        std::vector<double> result_sequential(size);
        std::vector<double> result_parallel(size);
        std::vector<double> correctResult(size);

        //start time for sequential gemv calculation
        auto start_time = std::chrono::high_resolution_clock::now();

        //Multiplication
        for (int i = 0; i < size; i++) {
            temp = 0;
            for (int j = 0; j < size; j++) {
                temp += matrix[i * size + j] * vector[j];
            }
            result_sequential[i] = temp;
        }

        //end time for gemv calculation
        auto end_time = std::chrono::high_resolution_clock::now();
        auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        //start time for parallel gemv calculation
        start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < size; i++) {
            futures.push_back(std::async(std::launch::async, process_row, i, size, std::ref(vector), std::ref(matrix)));
        }

        for (int i = 0; i < size; i++) {
            result_parallel[i] = futures[i].get();
        }

        //end time for gemv calculation
        end_time = std::chrono::high_resolution_clock::now();
        auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        //Checking result against BLAS implementation and calculating residual
        start_time = std::chrono::high_resolution_clock::now();
        cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, matrix.data(), size, vector.data(), 1, 1.0, correctResult.data(), 1);
        end_time = std::chrono::high_resolution_clock::now();
        auto mkl_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        double residual_sequential = calculateResidual(result_sequential , correctResult);
        double residual_parallel = calculateResidual(result_parallel, correctResult);

        double flops_sequential = (flops * 1000000) / sequential_duration; //Dividing by 1000000 to convert to seconds
        double flops_parallel = (flops * 1000000) / parallel_duration; //Dividing by 1000000 to convert to seconds
        double flops_mkl = (flops * 1000000) / mkl_duration; //Dividing by 1000000 to convert to seconds

        std::cout << "Size: " << size << std::endl;
        std::cout << "Sequential Implementation:" << std::endl;
        std::cout << "Time taken: " << sequential_duration << " microseconds" << std::endl;
        std::cout << "Flops: " << flops_sequential << std::endl;
        std::cout << "Residual: " << residual_sequential << "\n" << std::endl;
        std::cout << "Parallel Implementation:" << std::endl;
        std::cout << "Time taken: " << parallel_duration << " microseconds" << std::endl;
        std::cout << "Flops: " << flops_parallel << std::endl;
        std::cout << "Residual: " << residual_parallel << "\n" << std::endl;
        std::cout << "BLAS Implementation:" << std::endl;
        std::cout << "Time taken: " << mkl_duration << " microseconds" << std::endl;
        std::cout << "Flops: " << flops_mkl << std::endl;
        std::cout << "****************************\n" << std::endl;

        //Writing result into csv file
        if (outfile.is_open()) {
            outfile << size << "," << sequential_duration << "," << flops_sequential << "," << residual_sequential << "," << parallel_duration << "," << flops_parallel << "," << residual_parallel << "," << mkl_duration << "," << flops_mkl << std::endl;
        }
    }
}

//int main() {
//    std::cout << "Parallel Implementation" << std::endl;
//    std::vector<int> sizeList = { 10, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000 };
//    assignmentSolution(std::ref(sizeList));
//    return 0;
//}

static void printMatrix(std::vector<double> matrix, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << "[" << " ";
        for (int j = 0; j < n; j++) {
            std::cout << matrix[i * n + j] << ", ";
        }
        std::cout << "]" << " ";
        std::cout << std::endl;
    }
}

static void printVector(std::vector<double> vector) {
    std::cout << "[" << " ";
    for (double element : vector) {
        std::cout << element << ", ";
    }
    std::cout << "]" << " ";
    std::cout << std::endl;
}