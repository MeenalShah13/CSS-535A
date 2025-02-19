//#include <iostream>
//#include <vector>
//#include <random>
//#include <chrono>
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
//int main() {
//    const int size = 3;
//    double temp;
//
//    std::vector<double> matrix1 = { 1,1,1,-1,-1,-1,2,3,4 };
//    std::vector<double> matrix2 = { 1,2,3,0,0,0,6,7,8 };
//    std::vector<double> result(size * size);
//
//   auto start = std::chrono::high_resolution_clock::now();
//
//   //Multiplication
//   for (int i = 0; i<size; i++) {
//       for (int j = 0; j<size; j++) {
//           temp = 0;
//           for (int k = 0; k < size; k++) {
//               temp += matrix1[i*size + k] * matrix2[k*size + j];
//           }
//           result[i*size + j] = temp;
//       }
//   }
//
//   auto end = std::chrono::high_resolution_clock::now();
//
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//
//   std::cout << "Sequential Implementation for GEMM:\n" << std::endl;
//   std::cout << "Time taken: " << duration.count() << " microseconds\n" << std::endl;
//
//   std::cout << "Matrix1:" << std::endl;
//   printMatrix(std::ref(matrix1));
//   std::cout << "Matrix2:" << std::endl;
//   printMatrix(std::ref(matrix2));
//   std::cout << "Result of multiplication:" << std::endl;
//   printMatrix(std::ref(result));
//
//   return 0;
//}