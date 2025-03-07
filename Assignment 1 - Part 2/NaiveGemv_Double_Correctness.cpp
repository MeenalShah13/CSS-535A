//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
//
//std::vector<std::vector<double>> generateRandomMatrix(int n) {
//   // Seed the random number generator
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<> distrib(-10, 10);
//
//   // Create the matrix
//   std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
//
//   // Fill the matrix with random values
//   for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < n; ++j) {
//           matrix[i][j] = distrib(gen);
//       }
//   }
//
//   return matrix;
//}
//
//std::vector<double> generateRandomVector(int n) {
//   // Seed the random number generator
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<> distrib(-10, 10);
//
//   // Create a vector of the specified size
//   std::vector<double> randomVector(n);
//
//   // Fill the vector with random integers
//   for (int i = 0; i < n; ++i) {
//       randomVector[i] = distrib(gen);
//   }
//
//   return randomVector;
//}
//
//void printMatrix(double matrix[3][3]) {
//   for (int i=0; i<3; i++) {
//       std::cout << "[" << " ";
//       for (int j=0; j<3; j++) {
//           std::cout << matrix[i][j] << ", ";
//       }
//       std::cout << "]" << " ";
//       std::cout << std::endl;
//   }
//}
//
//void printVector(std::vector<double> vector) {
//   std::cout << "[" << " ";
//   for (double element : vector) {
//       std::cout << element << ", ";
//   }
//   std::cout << "]" << " ";
//   std::cout << std::endl;
//}
//
//int main() {
//   int size = 3;
//   double temp;
//
//   std::vector<double> vector = {1, 2, 3};
//   double matrix[3][3] = {1,2,3,0,0,0,6,7,8};
//   std::vector<double> result(size);
//
//   auto start = std::chrono::high_resolution_clock::now();
//
//   //Multiplication
//   for (int i = 0; i<size; i++) {
//       temp = 0;
//       for (int j = 0; j<size; j++) {
//           temp += matrix[i][j]*vector[j];
//       }
//       result[i] = temp;
//   }
//
//   auto end = std::chrono::high_resolution_clock::now();
//
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//   std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
//
//   std::cout << "Matrix:" << std::endl;
//   printMatrix(matrix);
//   std::cout << "Vector:" << std::endl;
//   printVector(vector);
//   std::cout << "Result of multiplication:" << std::endl;
//   printVector(result);
//
//   return 0;
//}