# C++ Digit Recognizer

This repository contains my implementation of a 3-layer neural network to predict the digit drawn in a 28x28 pixel canvas, using the famous MNIST dataset to train and test my model. My implementation does not use any pre-made machine learning or matrix libraries since I wanted to gain a complete understanding of how these basic neural networks operate. This project is an optimized version of my previous implementation: https://github.com/alexroz25/cpp_ml.

There is for sure some more room for improvement, but here's a list of some of the improvements I made:
- Utilize all CPU threads via OpenMP
- Implemented in-place transposition multiplication in the Matrix class
- Refactored Network class to eat less memory
- Significantly more modular
  - Added the ability to configure any number of layers of any number of neurons in the feed-forward neural network
- Improved Matrix algorithms to better take advantage of cache spatial locality
- Revisited the backpropagation algorithm and resolved issues with vanishing/exploding gradients
- Implemented leaky ReLU activation layers

These improvements resulted in a 90% speed-up in my testing. I also saw a massive improvement in the maximum learning rate before the model breaks down.

## Instructions

1. Download source files
2. Download mnist_train.csv and mnist_test.csv from the MNIST dataset source below. Append mnist_test.csv to mnist_train.csv (or just edit main.cpp to create a train/test split within mnist_train.csv).
3. Configure parameters at the top of main.cpp; hyperparameters in Matrix.h and Network.h.
4. Run main.cpp with c++17 and -fopenmp using input redirection
  - I used the following for compilation in VSCode:
  - g++ -std=c++17 -fopenmp -lpthread -O3 -o main main.cpp Matrix.h Network.h
  - ./main < mnist_train_test.csv

## MNIST dataset source: 

- https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
- https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view

## Main Resources Used:

- https://youtu.be/sIX_9n-1UbM
- https://youtu.be/w8yWXqWQYmU
- https://youtu.be/dB-u77Y5a6A

