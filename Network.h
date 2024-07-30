#ifndef NETWORK_H
#define NETWORK_H

#include "Matrix.h"
#include <random>
#include <cmath>
#include <string>

#define RANDOM_SEED 0 // type rd() for pseudo random seed num

class Network {
public:
    int dataPoints; // number of train/test examples
    std::vector<int> layerSizes;
    Matrix<float> target; // last layer neurons x dataPoints
    float LEARNING_RATE;
    // W.multiply(A)
    std::vector<std::vector<float>> biases; // 1 x layer size, 1 indexed
    Matrix<float> costs; // same size as last aLayers, compare against correctLabels
    std::vector<Matrix<float>> deltas; // computed for each neuron, neurons x samples 
    std::vector<Matrix<float>> zLayers; // input = [0], neurons x samples doesnt need an input layer though, 0/1 indexed
    std::vector<Matrix<float>> aLayers; // input = [0], neurons x samples, 0-indexed
    std::vector<Matrix<float>> weights; // weights[1] = weights applied to input layer, next x prev (aLayers[n] neurons x aLayers[n-1] neurons), 1 indexed

    Network(float LEARNING_RATE, int dataPoints, std::vector<int> layerSizes) : 
        dataPoints(dataPoints),
        layerSizes(layerSizes),
        LEARNING_RATE(LEARNING_RATE)
        {
            zLayers.emplace_back();
            aLayers.emplace_back(layerSizes[0], dataPoints);
            deltas.emplace_back();
            for (int i = 1; i < layerSizes.size(); ++i) {
                zLayers.emplace_back(layerSizes[i], dataPoints);
                aLayers.emplace_back(layerSizes[i], dataPoints);
                deltas.emplace_back(layerSizes[i], dataPoints);
            }
            initialize_random_weights_and_biases();
        }

    void initialize_random_weights_and_biases() {
        std::random_device rd;
        std::default_random_engine dre(RANDOM_SEED);
        std::normal_distribution<float> dist(0.0, 1.0);
        
        // initialize weight matrices
        weights.emplace_back();
        biases.emplace_back();

        int in = layerSizes.size();
        for (int i = 1; i < in; ++i) {
            
            int n = layerSizes[i] * layerSizes[i-1];
            std::vector<float> wlayer(n);

            // https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
            for (int j = 0; j < n; ++j) wlayer[j] = dist(dre) * sqrt(2.0/layerSizes[i-1]);
            
            weights.emplace_back(layerSizes[i], layerSizes[i-1], wlayer);
            biases.emplace_back(layerSizes[i]);
        }
    }

    void forward_propagate() {
        for (int i = 1; i < layerSizes.size()-1; ++i) {
            zLayers[i] = weights[i].multiply(aLayers[i-1]);
            aLayers[i] = zLayers[i].leaky_ReLU();
        }
        int last = layerSizes.size()-1;
        zLayers[last] = weights[last].multiply(aLayers[last-1]);
        aLayers[last] = zLayers[last].softmax();
    }

    float calculate_cost() {
        // https://www.stackoverflow.com/questions/57631507/how-can-i-take-the-derivative-of-the-softmax-output-in-back-prop
        int last = layerSizes.size()-1;
        deltas[last] = target.subtract(aLayers[last]);
        costs = deltas[last].square();

        // calculate MMSE
        int col = costs.cols(), row = costs.rows();
        float sumOfCosts = 0;
        int lastSize = layerSizes[last];
        for (int c = 0; c < col; ++c) {
            float sumOfIndividual = 0;
            for (int r = 0; r < row; ++r) {
                sumOfIndividual += costs.at(r, c);
            }
            sumOfCosts += sumOfIndividual / lastSize;
        }
        return sumOfCosts / dataPoints;
    }

    float calculate_accuracy() {
        int last = layerSizes.size()-1, lastSize = layerSizes[last];

        int correct = 0;
        for (int c = 0; c < dataPoints; ++c) {
            int correctLabel = -1, maxdex = -1;
            float maxVal = -1;
            for (int r = 0; r < lastSize; ++r) {
                if (target.at(r, c) > 0.5) correctLabel = r;
                if (aLayers[last].at(r, c) > maxVal) {
                    maxVal = aLayers[last].at(r, c);
                    maxdex = r;
                }
            }
            if (maxdex == correctLabel) ++correct;
        }
        return (float)correct / dataPoints;
    }

    void calculate_deltas() {
        int lastdex = layerSizes.size()-1;
        int lastSize = layerSizes[lastdex];

        for (int i = lastdex - 1; i > 0; --i) {
            deltas[i] = weights[i+1].transpose_multiply(deltas[i+1]).leaky_ReLU_derivative(zLayers[i]);
        }
    }

    void descend_gradient() {
        int lastdex = layerSizes.size()-1;
        for (int i = lastdex; i > 0; --i) {
            Matrix<float> dW = deltas[i].multiply_transpose(aLayers[i-1]).multiply(LEARNING_RATE / dataPoints);
            weights[i] = weights[i].add(dW);

            std::vector<float> dB = deltas[i].collapse();
            for (int j = 0; j < dB.size(); ++j) {
                dB[j] /= dataPoints;
                biases[i][j] += LEARNING_RATE * dB[j];
            }
        }
        // weights[2] = deltas[2] @ aLayers[1]^T
        // 2 x 60000 @ 60000 x 3 == 2 x 3
    }

    void read_csv() { // populate aLayers[0]
        
        int dataWidth = layerSizes[0];
        int lastSize = layerSizes[layerSizes.size()-1];
        target = Matrix<float>(lastSize, dataPoints);
        for (int i = 0; i < dataPoints; ++i) {
            std::string pix;
            std::getline(std::cin, pix, ',');
            int correctLabel = std::stoi(pix);
            
            for (int r = 0; r < lastSize; ++r) {
                if (r == correctLabel) target.at(r, i) = 1;
            }

            for (int j = 0; j < dataWidth-1; ++j) {
                std::getline(std::cin, pix, ',');
                aLayers[0].at(j, i) = std::stoi(pix) / 255.0;
            }
            std::getline(std::cin, pix, '\n');
            aLayers[0].at(dataWidth-1, i) = std::stoi(pix) / 255.0;
        }
        
    }
};


#endif