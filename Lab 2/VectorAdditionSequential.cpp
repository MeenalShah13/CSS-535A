//#include <iostream>
//#include <random>
//
//double* generateRandomVector(double* randomVector, int n) {
//    // Seed the random number generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> distrib(-10, 10);
//
//    // Fill the vector with random integers
//    for (int i = 0; i < n; ++i) {
//        randomVector[i] = distrib(gen);
//    }
//
//    return randomVector;
//}
//
//void printVector(double* vector, int n) {
//    std::cout << "[" << " ";
//    for (int i=0; i<n; i++) {
//        std::cout << vector[i] << ", ";
//    }
//    std::cout << "]" << " ";
//    std::cout << std::endl;
//}
//
//int main() {
//    int size = 10 * sizeof(double);
//    double* a = (double *) malloc(size);
//    generateRandomVector(a, 10);
//    double* b = (double *) malloc(size);
//    generateRandomVector(b, 10);
//    double* c = (double *) malloc(size);
//
//    for (int i = 0; i<10; i++) {
//        c[i] = a[i] + b[i];
//    }
//
//    std::cout << "Matrix a: ";
//    printVector(a, 10);
//    std::cout << "Matrix b: ";
//    printVector(b, 10);
//    std::cout << "Matrix c: ";
//    printVector(c, 10);
//
//    free(a);
//    free(b);
//    free(c);
//
//    return 0;
//}