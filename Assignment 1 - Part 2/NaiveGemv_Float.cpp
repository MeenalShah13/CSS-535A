//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//
//std::vector<std::vector<float>> generateRandomMatrix(int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Create the matrix
//    std::vector<std::vector<float>> matrix(n, std::vector<float>(n));
//
//    // Fill the matrix with random values
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            matrix[i][j] = distrib(gen);
//        }
//    }
//
//    return matrix;
//}
//
//std::vector<float> generateRandomVector(int n) {
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
//void printMatrix(std::vector<std::vector<float>> matrix) {
//    for (const auto& row : matrix) {
//        std::cout << "[" << " ";
//        for (float val : row) {
//            std::cout << val << ", ";
//        }
//        std::cout << "]" << " ";
//        std::cout << std::endl;
//    }
//}
//
//void printVector(std::vector<float> vector) {
//    std::cout << "[" << " ";
//    for (float element : vector) {
//        std::cout << element << ", ";
//    }
//    std::cout << "]" << " ";
//    std::cout << std::endl;
//}
//
//int main() {
//    int size = 100;
//    float temp;
//
//    std::vector<float> vector = generateRandomVector(size);
//    std::vector<std::vector<float>> matrix = generateRandomMatrix(size);
//    std::vector<float> result(size);
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    //Multiplication
//    for (int i = 0; i<size; i++) {
//        temp = 0;
//        for (int j = 0; j<size; j++) {
//            temp += matrix[i][j]*vector[j];
//        }
//        result[i] = temp;
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
//
//    // std::cout << "Matrix:" << std::endl;
//    // printMatrix(matrix);
//    // std::cout << "Vector:" << std::endl;
//    // printVector(vector);
//    // std::cout << "Result of multiplication:" << std::endl;
//    // printVector(result);
//
//    return 0;
//}