//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//#include <future>
//#include <mkl.h>
//#include <fstream>
//#include <cmath>
//
//std::vector<double> generateRandomMatrix(int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Create the matrix
//    std::vector<double> matrix(n * n);
//
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            matrix[i * n + j] = distrib(gen); // Row-major indexing as C++ is row-major
//        }
//    }
//
//    return matrix;
//}
//
//double calculateResidual(std::vector<double> calculatedResult, std::vector<double> blasMatrix) {
//    int size = std::sqrt(calculatedResult.size());
//    double sum = 0.0;
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            sum += std::abs(calculatedResult[i * size + j] - blasMatrix[i * size + j]);
//        }
//    }
//    return sum / (size * size);
//}
//
//// Non-Optimized matrix-vector multiplication function
//void process_row(int row_number, int col_number, int size, const std::vector<double>& matrix1, const std::vector<double>& matrix2, std::vector<double>& result) {
//    double temp = 0.0;
//    for (int k = 0; k < size; k++) {
//        temp += matrix1[row_number * size + k] * matrix2[k * size + col_number];
//    }
//    result[row_number * size + col_number] = temp;
//}
//
//// Optimized matrix-matrix multiplication function
//void process_stepped_row(int row_number, int size, const std::vector<double>& matrix1, const std::vector<double>& matrix2, std::vector<double>& result) {
//    double temp, temp1, temp2, temp3;
//    for (int i = 0; i < size; i++) {
//        temp = 0;
//        temp1 = 0;
//        temp2 = 0;
//        temp3 = 0;
//        for (int k = 0; k < size; k++) {
//            temp += matrix1[row_number * size + k] * matrix2[k * size + i];
//            temp1 += matrix1[(row_number + 1) * size + k] * matrix2[k * size + i];
//            temp2 += matrix1[(row_number + 2) * size + k] * matrix2[k * size + i];
//            temp3 += matrix1[(row_number + 3) * size + k] * matrix2[k * size + i];
//        }
//        result[row_number * size + i] = temp;
//        result[(row_number + 1) * size + i] = temp1;
//        result[(row_number + 2) * size + i] = temp2;
//        result[(row_number + 3) * size + i] = temp3;
//    }
//}
//
//void assignmentSolution(std::vector<int>& sizeList) {
//    //Handling writing results into a csv
//    std::ofstream outfile("Data.csv", std::ios::app);
//    if (!outfile.is_open()) {
//        std::cerr << "Error opening file: Data.csv" << std::endl;
//    }
//    std::vector<std::string> header = {"Size", "Theoretical Peak/Cycle", "Theoretical Peak", "Sequential Time in milliseconds", "Sequential Flops", "Sequential Efficiency", "Sequential Residual", "Parallel Time in milliseconds", "Parallel Flops", "Parallel Efficiency", "Parallel Residual", "BLAS Time in milliseconds", "BLAS Flops", "BLAS Efficiency"};
//    for (size_t i = 0; i < header.size(); ++i) {
//        outfile << header[i];
//        if (i < header.size() - 1) outfile << ","; // Comma separator
//    }
//    outfile << std::endl; // New line for header
//
//    //main program
//    for (int size : sizeList) {
//        int stepSize = 4;
//        double flop = 2 * std::pow(size, 3);
//        double memorySteps = std::pow(size, 3) + 3 * std::pow(size, 2);
//        double theoreticalPeakPerCycle = flop / memorySteps;
//        double theoreticalPeak = theoreticalPeakPerCycle * 16 * 2.2 * 1000000000; // theoreticalPeakPerCycle * numberOfCores * ClockSpeed
//        std::vector<std::future<void>> futures;
//        double temp;
//
//        std::vector<double> matrix1 = generateRandomMatrix(size);
//        std::vector<double> matrix2 = generateRandomMatrix(size);
//        std::vector<double> result_sequential(size * size);
//        std::vector<double> result_parallel(size * size);
//        std::vector<double> correctResult(size * size);
//
//        //start time for sequential gemv calculation
//        auto start_time = std::chrono::high_resolution_clock::now();
//
//        //Multiplication
//        for (int i = 0; i<size; i++) {
//            for (int j = 0; j<size; j++) {
//                temp = 0;
//                for (int k = 0; k < size; k++) {
//                    temp += matrix1[i * size + k] * matrix2[k * size + j];
//                }
//                result_sequential[i * size + j] = temp;
//            }
//        }
//
//        //end time for gemv calculation
//        auto end_time = std::chrono::high_resolution_clock::now();
//        auto sequential_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//
//        int i = 0, j;
//
//        //start time for parallel gemv calculation
//        start_time = std::chrono::high_resolution_clock::now();
//
//        while ((i + stepSize) < size) {
//            futures.push_back(std::async(std::launch::async, process_stepped_row, i, size, std::ref(matrix1), std::ref(matrix2), std::ref(result_parallel)));
//            i += stepSize;
//        }
//
//        while (i < size) {
//            j = 0;
//            while (j < size) {
//                futures.push_back(std::async(std::launch::async, process_row, i, j, size, std::ref(matrix1), std::ref(matrix2), std::ref(result_parallel)));
//                j++;
//            }
//            i++;
//        }
//
//        int threadSize = futures.size();
//        for (int i = 0; i < threadSize; i++) {
//            futures[i].get();
//        }
//
//        //end time for gemv calculation
//        end_time = std::chrono::high_resolution_clock::now();
//        auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//
//        //Checking result against BLAS implementation and calculating residual
//        start_time = std::chrono::high_resolution_clock::now();
//        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, matrix1.data(), size, matrix2.data(), size, 0.0, correctResult.data(), size);
//        end_time = std::chrono::high_resolution_clock::now();
//        auto mkl_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//
//        double residual_sequential = calculateResidual(result_sequential, correctResult);
//        double residual_parallel = calculateResidual(result_parallel, correctResult);
//
//        double flops_sequential = (flop * 1000) / sequential_duration; //Dividing by 1000000 to convert to seconds
//        double flops_parallel = (flop * 1000) / parallel_duration; //Dividing by 1000000 to convert to seconds
//        double flops_mkl = (flop * 1000) / mkl_duration; //Dividing by 1000000 to convert to seconds
//
//        double efficiency_sequential = flops_sequential / theoreticalPeak * 100;
//        double efficiency_parallel = flops_parallel / theoreticalPeak * 100;
//        double efficiency_mkl = flops_mkl / theoreticalPeak * 100;
//
//        std::cout << "Size: " << size << std::endl;
//        std::cout << "Sequential Implementation:" << std::endl;
//        std::cout << "Time taken: " << sequential_duration << " milliseconds" << std::endl;
//        std::cout << "Flops: " << flops_sequential << std::endl;
//        std::cout << "Efficiency: " << efficiency_sequential << std::endl;
//        std::cout << "Residual: " << residual_sequential << "\n" << std::endl;
//        std::cout << "Parallel Implementation:" << std::endl;
//        std::cout << "Time taken: " << parallel_duration << " milliseconds" << std::endl;
//        std::cout << "Flops: " << flops_parallel << std::endl;
//        std::cout << "Efficiency: " << efficiency_parallel << std::endl;
//        std::cout << "Residual: " << residual_parallel << "\n" << std::endl;
//        std::cout << "BLAS Implementation:" << std::endl;
//        std::cout << "Time taken: " << mkl_duration << " milliseconds" << std::endl;
//        std::cout << "Flops: " << flops_mkl << std::endl;
//        std::cout << "Efficiency: " << efficiency_mkl << std::endl;
//        std::cout << "****************************\n" << std::endl;
//
//        //Writing result into csv file
//        if (outfile.is_open()) {
//            outfile << size << "," << theoreticalPeakPerCycle << "," << theoreticalPeak << "," << sequential_duration << "," << flops_sequential << "," << efficiency_sequential << "," << residual_sequential << "," << parallel_duration << "," << flops_parallel << "," << efficiency_parallel << "," << residual_parallel << "," << mkl_duration << "," << flops_mkl << "," << efficiency_mkl << std::endl;
//        }
//    }
//}
//
//int main() {
//    //std::vector<int> sizeList = { 2, 3, 4 };
//    std::vector<int> sizeList = {10, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000 };
//    assignmentSolution(std::ref(sizeList));
//    return 0;
//}