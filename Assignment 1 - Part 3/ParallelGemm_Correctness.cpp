//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//#include <future>
//
//void printMatrix(std::vector<double>& matrix) {
//   for (int i=0; i<3; i++) {
//       std::cout << "[" << " ";
//       for (int j=0; j<3; j++) {
//           std::cout << matrix[i*3 + j] << ", ";
//       }
//       std::cout << "]" << " ";
//       std::cout << std::endl;
//   }
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
//    double temp, temp1;
//    for (int i = 0; i < size; i++) {
//        temp = 0;
//        temp1 = 0;
//        for (int k = 0; k < size; k++) {
//            temp += matrix1[row_number * size + k] * matrix2[k * size + i];
//            temp1 += matrix1[(row_number + 1) * size + k] * matrix2[k * size + i];
//        }
//        result[row_number * size + i] = temp;
//        result[(row_number + 1) * size + i] = temp1;
//    }
//}
//
//int main() {
//    int size = 3;
//    int stepSize = 2;
//    /*std::vector<std::future<double>> futures;*/
//    std::vector<std::future<void>> futures;
//
//    std::vector<double> matrix1 = { 1, 1, 1, -1, -1, -1, 2, 3, 4 };
//    std::vector<double> matrix2 = { 1, 2, 3, 0, 0, 0, 6, 7, 8 };
//    std::vector<double> result(size * size);
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    int i = 0;
//    while ((i+stepSize) < size) {
//        futures.push_back(std::async(std::launch::async, process_stepped_row, i, size, std::ref(matrix1), std::ref(matrix2), std::ref(result)));
//        i += stepSize;
//    }
//
//    int j;
//    while (i < size) {
//        j = 0;
//        while (j < size) {
//            futures.push_back(std::async(std::launch::async, process_row, i, j, size, std::ref(matrix1), std::ref(matrix2), std::ref(result)));
//            j++;
//        }
//        i++;
//    }
//
//    int threadSize = futures.size();
//    for (int i = 0; i < threadSize; i++) {
//        futures[i].get();
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    
//    std::cout << "Parallel Implementation for GEMM:\n" << std::endl;
//    std::cout << "Time taken: " << duration.count() << " microseconds\n" << std::endl;
//
//    std::cout << "Matrix1:" << std::endl;
//    printMatrix(matrix1);
//    std::cout << "Matrix2:" << std::endl;
//    printMatrix(matrix2);
//    std::cout << "Result of multiplication:" << std::endl;
//    printMatrix(result);
//
//    return 0;
//}