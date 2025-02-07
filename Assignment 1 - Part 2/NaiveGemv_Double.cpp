//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//#include <mkl.h>
//
//static std::vector<double> generateRandomMatrix(int n) {
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
//std::vector<double> generateRandomVector(int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Create a vector of the specified size
//    std::vector<double> randomVector(n);
//
//    // Fill the vector with random integers
//    for (int i = 0; i < n; ++i) {
//        randomVector[i] = distrib(gen);
//    }
//
//    return randomVector;
//}
//
//static std::vector<float> generateRandomMatrixFloat(int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Create the matrix
//    std::vector<float> matrix(n * n);
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
//std::vector<float> generateRandomVectorFloat(int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Create a vector of the specified size
//    std::vector<float> randomVector(n);
//
//    // Fill the vector with random integers
//    for (int i = 0; i < n; ++i) {
//        randomVector[i] = distrib(gen);
//    }
//
//    return randomVector;
//}
//
//static double calculateResidual(std::vector<double> vector, std::vector<double> blasVector) {
//    int size = vector.size();
//    double sum = 0.0;
//    for (int i = 0; i < size; i++) {
//        sum += std::abs(vector[i] - blasVector[i]);
//    }
//    return sum / size;
//}
//
//void floatSolution() {
//    float temp;
//
//    std::vector<float> vector = generateRandomVectorFloat(100);
//    std::vector<float> matrix = generateRandomMatrixFloat(100);
//    std::vector<float> result_sequential(100);
//    
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    //Multiplication
//    for (int i = 0; i < 100; i++) {
//        temp = 0;
//        for (int j = 0; j < 100; j++) {
//            temp += matrix[i * 100 + j] * vector[j];
//        }
//        result_sequential[i] = temp;
//    }
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
//
//    std::cout << "Size: 100" << std::endl;
//    std::cout << "Sequential Single Precision Implementation:" << std::endl;
//    std::cout << "Time taken: " << sequential_duration << " microseconds" << std::endl;
//}
//
//void assignmentSolution(std::vector<int> sizeList) {
//    //main program
//    for (int size : sizeList) {
//        double flop = 2 * size * size; //Number of floating point operation as per size
//        double temp;
//
//        std::vector<double> vector = generateRandomVector(size);
//        std::vector<double> matrix = generateRandomMatrix(size);
//        std::vector<double> result_sequential(size);
//        std::vector<double> correctResult(size);
//
//        auto start_time = std::chrono::high_resolution_clock::now();
//
//        //Multiplication
//        for (int i = 0; i < size; i++) {
//            temp = 0;
//            for (int j = 0; j < size; j++) {
//                temp += matrix[i * size + j] * vector[j];
//            }
//            result_sequential[i] = temp;
//        }
//
//        auto end_time = std::chrono::high_resolution_clock::now();
//        auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
//
//        //Checking result against BLAS implementation and calculating residual
//        start_time = std::chrono::high_resolution_clock::now();
//        cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, matrix.data(), size, vector.data(), 1, 1.0, correctResult.data(), 1);
//        end_time = std::chrono::high_resolution_clock::now();
//        auto mkl_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
//
//        double residual_sequential = calculateResidual(result_sequential, correctResult);
//
//        double flops_sequential = (flop * 1000000) / sequential_duration; //Dividing by 1000000 to convert to seconds
//        double flops_mkl = (flop * 1000000) / mkl_duration; //Dividing by 1000000 to convert to seconds
//
//        std::cout << "Size: " << size << std::endl;
//        std::cout << "Sequential Implementation:" << std::endl;
//        std::cout << "Time taken: " << sequential_duration << " microseconds" << std::endl;
//        std::cout << "Flops: " << flops_sequential << std::endl;
//        std::cout << "Residual: " << residual_sequential << "\n" << std::endl;
//        std::cout << "BLAS Implementation:" << std::endl;
//        std::cout << "Time taken: " << mkl_duration << " microseconds" << std::endl;
//        std::cout << "Flops: " << flops_mkl << std::endl;
//        std::cout << "****************************\n" << std::endl;
//    }
//}
//
//int main() {
//    floatSolution();
//    std::cout << "Naive Sequential Double Precision Implementation" << std::endl;
//    std::vector<int> sizeList = { 10, 50, 100, 300, 400, 600, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000 };
//    assignmentSolution(sizeList);
//    return 0;
//}