#include<iostream>
#include<cfloat>

int main() {
    double macheps = 1.0;
    while(1 + macheps != 1) {
        macheps /= 2;
    }
    macheps *= 2;

    if(macheps == DBL_EPSILON){
        std::cout << "Correct Answer" << std::endl;
        std::cout << macheps << std::endl;
    } else {
        std::cout << "Wrong Answer. Actual Answer:" << std::endl;
        std::cout << DBL_EPSILON << std::endl;
        std::cout << "Your answer:" << std::endl;
        std::cout << macheps << std::endl;
    }

    return 0;
}