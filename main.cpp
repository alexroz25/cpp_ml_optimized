#include <iostream>
#include "Network.h"
#include "Matrix.h"

float LEARNING_RATE = 0.2;
int dataPoints = 60000;
int epochs = 100;
int infoInterval = 1;
std::vector<int> layerSizes = {784, 16, 10};
// #define UNIT_TEST
#define TIMER

#ifdef TIMER
    #include <chrono>
#endif
#ifdef UNIT_TEST
    #include "Test.h"
#endif

using namespace std;

void generate_tests(int n) {
    srand(time(0));

    for (int i = 0; i < n; ++i) {
        int a = rand() & 1, b = rand() & 1;
        cout << (a^b)  << " " << a << " " << b << endl;
    }
}

int main() {
    ios_base::sync_with_stdio(false);

#ifdef UNIT_TEST
	TEST();
#endif
    
#ifdef TIMER
    auto t1 = chrono::high_resolution_clock::now();
    Network train(LEARNING_RATE, dataPoints, layerSizes);
    train.read_csv();
    auto t2 = chrono::high_resolution_clock::now();
    cout << "Read " << dataPoints << " datapoints in " << chrono::duration<double>(t2 - t1).count() << "s.\n" << endl;
#else
    Network train(LEARNING_RATE, dataPoints, layerSizes);
    train.read_csv();
#endif

    cout << "Epoch,Average Cost,Accuracy" << endl;
    int n = 0;

#ifdef TIMER
    t1 = chrono::high_resolution_clock::now();
#endif
    while (n++ < epochs) {
        train.forward_propagate();
        float averageCost = train.calculate_cost();
        float accuracy = train.calculate_accuracy();
        train.calculate_deltas();
        train.descend_gradient();

        if (n % infoInterval == 0)
            cout << n << ',' << averageCost << ',' << accuracy << endl;
    } cout << endl;
    
#ifdef TIMER
    t2 = chrono::high_resolution_clock::now();
    cout << "Training Time: " << chrono::duration<double>(t2 - t1).count() << "s.\n" << endl;
#endif

    Network test(LEARNING_RATE, 10000, layerSizes);
    test.read_csv();
    test.biases = train.biases;
    test.weights = train.weights;
    test.forward_propagate();
    cout << "mnist_test Results:\nAverage Cost: " << test.calculate_cost() << " | Accuracy: " << test.calculate_accuracy() << endl;
    

    return 0;
}