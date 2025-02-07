//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//#include <future>
//#include <mkl.h>
//#include <iomanip>
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
//static std::vector<double> generateRandomVector(int n) {
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
//static void printMatrix(std::vector<double> matrix, int n) {
//    for (int i = 0; i < n; i++) {
//        std::cout << "[" << " ";
//        for (int j = 0; j < n; j++) {
//            std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1)  << matrix[i * n + j] << ", ";
//        }
//        std::cout << "]" << " ";
//        std::cout << std::endl;
//    }
//}
//
//static void printVector(std::vector<double> vector) {
//    std::cout << "[" << " ";
//    for (double element : vector) {
//        std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1) << element << ", ";
//    }
//    std::cout << "]" << " ";
//    std::cout << std::endl;
//}
//
//static double calculateResidual(std::vector<double> vector, std::vector<double> blasVector) {
//    int size = vector.size();
//    double sum = 0.0;
//    for (int i = 0; i < size; i++) {
//        sum += vector[i] - blasVector[i];
//    }
//    return sum / size;
//}
//
//// Optimized matrix-vector multiplication function
//static double process_row(int row_number, int size, const std::vector<double>& vector, const std::vector<double>& matrix) {
//    double temp = 0.0;
//    for (size_t j = 0; j < size; ++j) {
//        temp += vector[j] * matrix[row_number* size + j];
//    }
//    return temp;
//}
//
//int main() {
//    int size = 3;
//    std::vector<std::future<double>> futures;
//
//    std::vector<double> vector = { 1, 2, 3 };
//    std::vector<double> matrix = { 1,2,3,0,0,0,6,7,8 };
//    std::vector<double> result(size);
//    std::vector<double> correctResult(size);
//
//    //start time for gemv calculation
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    for (int i = 0; i < size; i++) {
//        futures.push_back(std::async(std::launch::async, process_row, i, size, std::ref(vector), std::ref(matrix)));
//    }
//
//    for (int i = 0; i < size; i++) {
//        result[i] = futures[i].get();
//    }
//
//    //end time for gemv calculation
//    auto end_time = std::chrono::high_resolution_clock::now();
//
//    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, matrix.data(), size, vector.data(), 1, 1.0, correctResult.data(), 1);
//    double residual = calculateResidual(result, correctResult);
//
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//    std::cout << "Size: " << size << std::endl;
//    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
//    std::cout << "Residual: " << residual << std::endl;
//    std::cout << "****************************\n" << std::endl;
//
//    std::cout << "Matrix:" << std::endl;
//    printMatrix(matrix, size);
//    std::cout << "Vector:" << std::endl;
//    printVector(vector);
//    std::cout << "Blas Vector:" << std::endl;
//    printVector(correctResult);
//    std::cout << "Result of multiplication:" << std::endl;
//    printVector(result);
//
//    return 0;
//}