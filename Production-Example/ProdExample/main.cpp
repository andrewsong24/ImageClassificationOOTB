#include <iostream>

int main(int argc, char *argv[]) {

    /**
     * argv[1] should be the model
     */

    if(argc < 2){
        std::cerr << "Model was not put as second command-line argument";
        return -1;
    }

    
    return 0;
}